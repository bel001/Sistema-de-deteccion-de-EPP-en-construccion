from __future__ import annotations

import csv
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from src.engine import EPPDetectionEngine as DetectionEngine, DetectionResult
from src.epp_utils import (
    DEFAULT_IOU,
    IMAGE_EXTENSIONS,
    SAFETY_VEST_CLASS_ID,
    VIDEO_DISPLAY_CLASS_IDS,
    WEBCAM_CLASS_IDS,
    analyze_compliance,
    detect_connected_cameras,
    draw_detection_boxes,
    draw_status_panel,
    draw_status_panel_big,
    filter_supported_class_ids,
    model_supports_class_id,
)


COLORS = {
    "bg": "#0d1117",
    "panel": "#161b22",
    "accent": "#1f6feb",
    "success": "#3fb950",
    "warning": "#d29922",
    "danger": "#f85149",
    "text": "#e6edf3",
    "text_dim": "#a1aab3",  # mejor contraste AAA sobre #161b22
    "border": "#30363d",
    "text_muted": "#8b949e",
}

FONTS = {
    "title": ("Segoe UI", 22, "bold"),
    "heading": ("Segoe UI", 14, "bold"),
    "body": ("Segoe UI", 11),
    "mono": ("Consolas", 11),
    "small": ("Segoe UI", 10),  # HiDPI
    "small_dim": ("Segoe UI", 9),
    "big_button": ("Segoe UI", 13, "bold"),
}

APP_VERSION = "v2.1"


def open_system_file_dialog(title: str, patterns: list[str]) -> str:
    """Abre el selector de archivos del sistema cuando esta disponible."""
    zenity = shutil.which("zenity")
    if zenity is not None:
        cmd = [zenity, "--file-selection", "--title", title]
        if patterns:
            cmd.extend(["--file-filter", "Archivos soportados | " + " ".join(patterns)])
        result = subprocess.run(cmd, text=True, capture_output=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else ""

    kdialog = shutil.which("kdialog")
    if kdialog is not None:
        filter_text = " ".join(patterns) if patterns else "*"
        result = subprocess.run(
            [kdialog, "--getopenfilename", str(Path.home()), filter_text],
            text=True,
            capture_output=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    return filedialog.askopenfilename(
        title=title,
        filetypes=[("Archivos soportados", " ".join(patterns)), ("Todos", "*.*")],
    )


def open_system_directory_dialog(title: str) -> str:
    """Abre el selector de carpetas del sistema cuando esta disponible."""
    zenity = shutil.which("zenity")
    if zenity is not None:
        result = subprocess.run(
            [zenity, "--file-selection", "--directory", "--title", title],
            text=True,
            capture_output=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    kdialog = shutil.which("kdialog")
    if kdialog is not None:
        result = subprocess.run(
            [kdialog, "--getexistingdirectory", str(Path.home())],
            text=True,
            capture_output=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    return filedialog.askdirectory(title=title)


def panel_label(parent: tk.Widget, text: str, font_key: str = "heading",
                fg_key: str = "accent") -> tk.Label:
    label = tk.Label(parent, text=text, bg=COLORS["panel"], fg=COLORS[fg_key],
                     font=FONTS[font_key])
    label.pack(pady=(20, 10))
    return label


def action_button(parent: tk.Widget, text: str, color_key: str, command,
                  state: Literal["normal", "active", "disabled"] = "normal") -> tk.Button:
    button = tk.Button(parent, text=text, bg=COLORS[color_key], fg=COLORS["text"],
                       font=FONTS["big_button"], bd=0,
                       activebackground=COLORS["border"], cursor="hand2",
                       state=state, command=command)
    button.pack(pady=10)
    return button


def result_text(parent: tk.Widget, height: int, initial: str = "") -> tk.Text:
    text = tk.Text(parent, bg=COLORS["bg"], fg=COLORS["text"],
                   font=FONTS["mono"], bd=0, height=height,
                   highlightbackground=COLORS["border"], highlightthickness=1)
    text.pack(fill="both", expand=True, padx=12, pady=12)
    if initial:
        text.insert("1.0", initial)
    return text


def replace_text(widget: tk.Text, content: str) -> None:
    widget.delete("1.0", "end")
    widget.insert("1.0", content)


def two_column_body(parent: tk.Widget, right_width: int = 350,
                    right_fill: Literal["none", "x", "y", "both"] = "y",
                    right_expand: bool = False) -> tuple[tk.Frame, tk.Frame]:
    body = tk.Frame(parent, bg=COLORS["bg"])
    body.pack(fill="both", expand=True, padx=8, pady=8)

    left = tk.Frame(body, bg=COLORS["panel"])
    left.pack(side="left", fill="both", expand=True)

    right = tk.Frame(body, bg=COLORS["panel"], width=right_width)
    right.pack(side="right", fill=right_fill, expand=right_expand, padx=(8, 0))
    right.pack_propagate(False)
    return left, right


def photo_from_frame(label: tk.Widget, frame: np.ndarray,
                     fallback_size: tuple[int, int]) -> ImageTk.PhotoImage:
    # Fallback robusto: en init winfo es 1; usar tamaño real disponible con clamp
    cw = label.winfo_width()
    ch = label.winfo_height()
    if cw < 50 or ch < 50:
        cw, ch = fallback_size
    cw = max(120, min(cw, fallback_size[0] * 2))
    ch = max(90, min(ch, fallback_size[1] * 2))
    h, w = frame.shape[:2]
    scale = min(cw / w, ch / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img = Image.frombytes("RGB", (nw, nh), rgb.tobytes())
    return ImageTk.PhotoImage(image=img)


class CardButton(tk.Frame):
    def __init__(self, parent: tk.Widget, text: str, desc: str, icon: str,
                 color: str, command, **kwargs) -> None:
        super().__init__(parent, bg=COLORS["border"], cursor="hand2", **kwargs)
        self._command = command
        self._color = color
        self.pack_propagate(False)

        inner = tk.Frame(self, bg=COLORS["panel"])
        inner.pack(fill="both", expand=True, padx=2, pady=2)

        tk.Label(inner, text=icon, bg=COLORS["panel"], fg=color,
                 font=("Segoe UI", 40)).pack(pady=(16, 4))
        tk.Label(inner, text=text, bg=COLORS["panel"], fg=COLORS["text"],
                 font=FONTS["big_button"]).pack()
        tk.Label(inner, text=desc, bg=COLORS["panel"], fg=COLORS["text_dim"],
                 font=FONTS["small"]).pack(pady=(2, 16))

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", lambda e: self._command())
        inner.bind("<Enter>", self._on_enter)
        inner.bind("<Leave>", self._on_leave)
        inner.bind("<Button-1>", lambda e: self._command())
        for child in inner.winfo_children():
            child["cursor"] = "hand2"
            child.bind("<Enter>", self._on_enter)
            child.bind("<Leave>", self._on_leave)
            child.bind("<Button-1>", lambda e: self._command())

    def _on_enter(self, _=None) -> None:
        self.configure(bg=self._color)

    def _on_leave(self, _=None) -> None:
        self.configure(bg=COLORS["border"])




class MainApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Sistema de Detección de EPP")
        self.configure(bg=COLORS["bg"])
        self.geometry("1200x750")
        self.minsize(900, 600)
        self.resizable(True, True)

        self._container = tk.Frame(self, bg=COLORS["bg"])
        self._container.pack(fill="both", expand=True)
        self._current_frame: tk.Frame | None = None
        self._engine: DetectionEngine | None = None
        self._engine_loading = False
        self._engine_error: str | None = None

        self._setup_styles()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        # Atajos globales
        self.bind("<Escape>", lambda e: self._on_close())
        self.bind("<F11>", lambda e: self.attributes("-fullscreen", not self.attributes("-fullscreen")))
        self._show_frame(MainMenu)

    def _setup_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Title.TLabel", background=COLORS["bg"], foreground=COLORS["text"],
                        font=FONTS["title"])
        style.configure("Heading.TLabel", background=COLORS["bg"], foreground=COLORS["text"],
                        font=FONTS["heading"])
        style.configure("Body.TLabel", background=COLORS["bg"], foreground=COLORS["text_dim"],
                        font=FONTS["body"])
        style.configure("Value.TLabel", background=COLORS["panel"], foreground=COLORS["text"],
                        font=FONTS["body"])
        style.configure("StatusBar.TLabel", background=COLORS["panel"], foreground=COLORS["text_dim"],
                        font=FONTS["small"])

    def _show_frame(self, frame_class: type, **kwargs) -> None:
        if self._current_frame is not None and self._current_frame.winfo_exists():
            cleanup = getattr(self._current_frame, "_cleanup", None)
            if callable(cleanup):
                cleanup()

        for w in self._container.winfo_children():
            w.destroy()

        frame = frame_class(self._container, self, **kwargs)
        if not frame.winfo_exists():
            return
        frame.pack(fill="both", expand=True)
        self._current_frame = frame

    def _on_close(self) -> None:
        self.withdraw()
        if self._current_frame is not None and self._current_frame.winfo_exists():
            cleanup = getattr(self._current_frame, "_cleanup", None)
            if callable(cleanup):
                cleanup()
        cv2.destroyAllWindows()
        self.destroy()

    def get_engine(self) -> "DetectionEngine":
        if self._engine is None:
            if self._engine_loading:
                raise RuntimeError("Modelo aún cargando, espera unos segundos...")
            if self._engine_error:
                raise RuntimeError(self._engine_error)
            self._engine = DetectionEngine()
        return self._engine

    def preload_engine_async(self, on_done=None, on_error=None) -> None:
        if self._engine is not None or self._engine_loading:
            if on_done:
                on_done()
            return
        self._engine_loading = True
        self._engine_error = None
        def _load():
            try:
                eng = DetectionEngine()
                def _ok():
                    self._engine = eng
                    self._engine_loading = False
                    if on_done:
                        on_done()
                self.after(0, _ok)
            except Exception as e:
                def _err():
                    self._engine_error = str(e)
                    self._engine_loading = False
                    if on_error:
                        on_error(str(e))
                self.after(0, _err)
        threading.Thread(target=_load, daemon=True).start()


class MainMenu(tk.Frame):
    def __init__(self, parent: tk.Widget, app: MainApp) -> None:
        super().__init__(parent, bg=COLORS["bg"])
        self.app = app

        header = tk.Frame(self, bg=COLORS["bg"])
        header.pack(fill="x", pady=(30, 6))
        tk.Label(header, text="Sistema de Detección de EPP",
                 bg=COLORS["bg"], fg=COLORS["text"], font=FONTS["title"]).pack()
        tk.Label(header, text="Equipo de Protección Personal — Construcción",
                 bg=COLORS["bg"], fg=COLORS["text_dim"], font=FONTS["body"]).pack()
        # Pre-carga modelo en segundo plano con indicador
        self._loading_label = tk.Label(header, text="Cargando modelo YOLO...", bg=COLORS["bg"],
                                       fg=COLORS["warning"], font=FONTS["small"])
        self._loading_label.pack(pady=(6,0))
        def _on_loaded():
            self._loading_label.configure(text="✓ Modelo listo", fg=COLORS["success"])
            self.after(1500, lambda: self._loading_label.pack_forget())
        def _on_err(msg):
            self._loading_label.configure(text=f"✗ Error modelo: {msg[:60]}", fg=COLORS["danger"])
        app.preload_engine_async(on_done=_on_loaded, on_error=_on_err)
        if app._engine is not None:
            _on_loaded()

        # Grid responsive de cards (2 columnas, se adapta a 900px)
        cards_outer = tk.Frame(self, bg=COLORS["bg"])
        cards_outer.pack(expand=True, fill="both", padx=20, pady=10)
        cards_outer.columnconfigure(0, weight=1)
        cards_outer.columnconfigure(1, weight=1)

        options = [
            ("Cámara en Vivo", "Tiempo real\n+ Dahua/RTSP", "🎥", "success", WebcamView),
            ("Video", "Archivo con\npreview embebido", "🎬", "accent", VideoView),
            ("Imagen", "Estática con\nsliders umbral", "🖼️", "warning", ImageView),
            ("Lote de Imágenes", "Carpeta + CSV\nsanitizado", "📁", "danger", BatchView),
        ]

        for idx, (title, desc, icon, color_key, view) in enumerate(options):
            r, c = divmod(idx, 2)
            cell = tk.Frame(cards_outer, bg=COLORS["bg"])
            cell.grid(row=r, column=c, padx=10, pady=10, sticky="nsew")
            CardButton(cell, title, desc, icon, COLORS[color_key],
                       lambda target=view: app._show_frame(target),
                       width=260, height=190).pack(expand=True)

        tk.Label(self, text="Selecciona un modo para comenzar  •  q/Esc para salir  •  F11 pantalla completa",
                 bg=COLORS["bg"], fg=COLORS["text_muted"], font=FONTS["small_dim"]).pack(pady=(4, 6))

        # Footer
        footer = tk.Frame(self, bg=COLORS["panel"], height=22)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)
        engine_info = app._engine.runtime_label if app._engine else ("cargando..." if app._engine_loading else "no cargado")
        tk.Label(footer, text=f"  {APP_VERSION}  •  {engine_info}  •  YOLOv8 SH17 10 clases",
                 bg=COLORS["panel"], fg=COLORS["text_dim"], font=FONTS["small_dim"], anchor="w").pack(side="left", padx=8)
        tk.Label(footer, text="Ayuda: README.md  •  EXPLICACION.md  ",
                 bg=COLORS["panel"], fg=COLORS["text_muted"], font=FONTS["small_dim"], anchor="e").pack(side="right", padx=8)


class BaseView(tk.Frame):
    def __init__(self, parent: tk.Widget, app: MainApp) -> None:
        super().__init__(parent, bg=COLORS["bg"])
        self.app = app

        top = tk.Frame(self, bg=COLORS["panel"], height=48)
        top.pack(fill="x")
        top.pack_propagate(False)

        tk.Button(top, text="◀ Volver al menú", bg=COLORS["panel"], fg=COLORS["accent"],
                  font=FONTS["body"], bd=0, activebackground=COLORS["border"],
                  activeforeground=COLORS["text"], cursor="hand2",
                  command=lambda: self._cleanup() or app._show_frame(MainMenu)
                  ).pack(side="left", padx=12, pady=8)

        self.status_label = tk.Label(top, text="", bg=COLORS["panel"],
                                     fg=COLORS["text_dim"], font=FONTS["small"])
        self.status_label.pack(side="right", padx=12, pady=8)

    def _cleanup(self) -> None:
        pass

    def _set_status(self, text: str) -> None:
        try:
            if not self.winfo_exists():
                return
            self.status_label.configure(text=text)
        except tk.TclError:
            return

    def _run_on_ui(self, callback) -> None:
        try:
            if not self.winfo_exists():
                return
        except tk.TclError:
            return

        def guarded_callback() -> None:
            try:
                if not self.winfo_exists():
                    return
                callback()
            except tk.TclError:
                return

        try:
            self.after(0, guarded_callback)
        except tk.TclError:
            return

    def _load_engine(self) -> DetectionEngine | None:
        try:
            return self.app.get_engine()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return None


class WebcamView(BaseView):
    def __init__(self, parent: tk.Widget, app: MainApp) -> None:
        super().__init__(parent, app)
        self._running = False
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._cap = None
        self._engine: DetectionEngine | None = None
        self._thread: threading.Thread | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._last_error_report_time = 0.0
        self._current_source: int | str = 0
        self._mirror_var = tk.BooleanVar(value=True)
        self._turbo_var = tk.BooleanVar(value=False)  # defecto OFF restaura detección 1:1
        self._frame_count_idx = 0

        self._report_data = {
            "fps": 0.0, "detections": [], "alerts": [],
            "unprotected_heads": 0, "persons": 0, "frames": 0,
            "start_time": 0.0, "has_vest_model": False,
            "runtime": "",
        }

        body = tk.Frame(self, bg=COLORS["bg"])
        body.pack(fill="both", expand=True, padx=8, pady=8)

        video_frame = tk.LabelFrame(body, text="VIDEO EN VIVO", bg=COLORS["panel"],
                                    fg=COLORS["accent"], font=FONTS["heading"],
                                    highlightbackground=COLORS["border"],
                                    highlightthickness=1, bd=0)
        video_frame.pack(side="left", fill="both", expand=True)

        # Barra de control de cámara en 2 filas para evitar overflow en 900px
        ctrl_bar = tk.Frame(video_frame, bg=COLORS["panel"])
        ctrl_bar.pack(fill="x", padx=8, pady=4)

        # Fila 1: selector fuente + RTSP/Dahua
        row1 = tk.Frame(ctrl_bar, bg=COLORS["panel"])
        row1.pack(fill="x", pady=2)
        tk.Label(row1, text="Fuente:", bg=COLORS["panel"],
                 fg=COLORS["text"], font=FONTS["small"]).pack(side="left", padx=(0, 4))

        self._camera_combo = ttk.Combobox(
            row1, values=[], state="readonly", width=24, font=FONTS["small"])
        self._camera_combo.pack(side="left", padx=4)
        self._camera_combo.bind("<<ComboboxSelected>>", self._on_combo_changed)

        tk.Button(row1, text="🔍", bg=COLORS["panel"], fg=COLORS["text"],
                  font=FONTS["small"], bd=1, cursor="hand2",
                  activebackground=COLORS["border"], command=self._scan_and_populate_cameras
                  ).pack(side="left", padx=2)

        self._dahua_preset = ttk.Combobox(
            row1, values=["Genérico RTSP", "Dahua subtype=0 (1080p)", "Dahua subtype=1 (480p WiFi)",
                          "Hikvision 101", "HTTP MJPEG"], state="readonly", width=22, font=FONTS["small"])
        self._dahua_preset.current(0)
        self._dahua_preset.pack(side="left", padx=6)
        self._dahua_preset.bind("<<ComboboxSelected>>", self._on_preset_changed)

        tk.Button(row1, text="▶ Conectar", bg=COLORS["success"], fg=COLORS["text"],
                  font=FONTS["small"], bd=0, cursor="hand2",
                  activebackground=COLORS["border"], command=self._switch_camera
                  ).pack(side="left", padx=6)

        # Fila 2: entrada URL + opciones
        row2 = tk.Frame(ctrl_bar, bg=COLORS["panel"])
        row2.pack(fill="x", pady=2)
        self._camera_entry = tk.Entry(row2, bg=COLORS["bg"], fg=COLORS["text"],
                                      font=FONTS["small"], bd=1,
                                      highlightbackground=COLORS["border"])
        self._camera_entry.insert(0, "rtsp://admin:pass@192.168.1.108:554/cam/realmonitor?channel=1&subtype=1")
        self._camera_entry.pack(side="left", fill="x", expand=True, padx=(0,6))

        tk.Checkbutton(row2, text="🪞 Espejo", variable=self._mirror_var,
                       bg=COLORS["panel"], fg=COLORS["text"],
                       selectcolor=COLORS["panel"], activebackground=COLORS["panel"],
                       activeforeground=COLORS["text"], font=FONTS["small"]
                       ).pack(side="left", padx=4)

        tk.Checkbutton(row2, text="⚡ Turbo", variable=self._turbo_var,
                       bg=COLORS["panel"], fg=COLORS["success"],
                       selectcolor=COLORS["panel"], activebackground=COLORS["panel"],
                       activeforeground=COLORS["success"], font=FONTS["small"]
                       ).pack(side="left", padx=4)

        # Fila 3: sliders confianza (compactos)
        row3 = tk.Frame(ctrl_bar, bg=COLORS["panel"])
        row3.pack(fill="x", pady=2)
        tk.Label(row3, text="Conf:", bg=COLORS["panel"], fg=COLORS["text_dim"], font=FONTS["small_dim"]).pack(side="left")
        self._webcam_conf_var = tk.DoubleVar(value=0.25)
        self._webcam_ppe_var = tk.DoubleVar(value=0.25)
        self._webcam_helmet_var = tk.DoubleVar(value=0.15)
        for label, var, to in [("Gral", self._webcam_conf_var, 0.5), ("EPP", self._webcam_ppe_var, 0.5), ("Casco", self._webcam_helmet_var, 0.4)]:
            tk.Label(row3, text=label, bg=COLORS["panel"], fg=COLORS["text_muted"], font=FONTS["small_dim"]).pack(side="left", padx=(8,2))
            tk.Scale(row3, variable=var, from_=0.05, to=to, resolution=0.05, orient="horizontal",
                     bg=COLORS["panel"], fg=COLORS["text_dim"], highlightthickness=0, length=70,
                     font=FONTS["small_dim"], troughcolor=COLORS["border"], activebackground=COLORS["accent"]
                     ).pack(side="left")

        self._canvas = tk.Label(video_frame, bg="#000000")
        self._canvas.pack(fill="both", expand=True, padx=4, pady=4)

        report_frame = tk.LabelFrame(body, text="REPORTE EN VIVO", bg=COLORS["panel"],
                                     fg=COLORS["accent"], font=FONTS["heading"],
                                     highlightbackground=COLORS["border"],
                                     highlightthickness=1, bd=0, width=340)
        report_frame.pack(side="right", fill="y", padx=(8, 0))
        report_frame.pack_propagate(False)

        self._report_canvas = tk.Canvas(report_frame, bg=COLORS["panel"],
                                        highlightthickness=0)
        self._report_scroll = tk.Scrollbar(report_frame, orient="vertical",
                                           command=self._report_canvas.yview)
        self._report_inner = tk.Frame(self._report_canvas, bg=COLORS["panel"])
        self._report_inner.bind("<Configure>",
                                lambda e: self._report_canvas.configure(
                                    scrollregion=self._report_canvas.bbox("all")))
        self._report_canvas.create_window((0, 0), window=self._report_inner, anchor="nw")
        self._report_canvas.configure(yscrollcommand=self._report_scroll.set)
        self._report_canvas.pack(side="left", fill="both", expand=True)
        self._report_scroll.pack(side="right", fill="y")

        self._build_report_widgets()
        self._scan_and_populate_cameras()
        self._start_webcam(source=self._current_source)

    def _scan_and_populate_cameras(self) -> None:
        self._detected_cams = detect_connected_cameras()
        labels = [label for _, label in self._detected_cams]
        self._camera_combo["values"] = labels
        if labels:
            self._camera_combo.current(0)
            self._current_source = self._detected_cams[0][0]
        self._on_combo_changed()

    def _on_preset_changed(self, event=None) -> None:
        preset = self._dahua_preset.get()
        base = self._camera_entry.get().strip()
        # Extraer IP si ya hay URL
        ip = "192.168.1.108"
        if "@" in base and "/" in base:
            try:
                ip = base.split("@")[1].split(":")[0].split("/")[0]
            except Exception:
                pass
        mapping = {
            "Dahua subtype=0 (1080p)": f"rtsp://admin:pass@{ip}:554/cam/realmonitor?channel=1&subtype=0",
            "Dahua subtype=1 (480p WiFi)": f"rtsp://admin:pass@{ip}:554/cam/realmonitor?channel=1&subtype=1",
            "Hikvision 101": f"rtsp://admin:pass@{ip}:554/Streaming/Channels/101",
            "HTTP MJPEG": f"http://{ip}:8080/video",
            "Genérico RTSP": base if base.startswith("rtsp") else f"rtsp://admin:pass@{ip}:554/stream1",
        }
        if preset in mapping:
            self._camera_entry.delete(0, "end")
            self._camera_entry.insert(0, mapping[preset])
            self._set_status(f"Preset {preset} → edita IP/pass y Conectar")

    def _on_combo_changed(self, event=None) -> None:
        idx = self._camera_combo.current()
        if idx >= 0 and idx < len(self._detected_cams):
            src_val, _ = self._detected_cams[idx]
            # Mantener entry siempre visible en diseño 2 filas (no pack_forget)
            if src_val == "rtsp":
                self._camera_entry.configure(state="normal")
            else:
                # Sugerir que puede sobrescribir con RTSP aun en cámara local
                pass

    def _switch_camera(self) -> None:
        idx = self._camera_combo.current()
        if idx >= 0 and idx < len(self._detected_cams):
            src_val, _ = self._detected_cams[idx]
            if src_val == "rtsp":
                custom = self._camera_entry.get().strip()
                if custom.isdigit():
                    source = int(custom)
                elif custom and custom != "rtsp://...":
                    source = custom
                else:
                    source = 0
            else:
                source = src_val
        else:
            source = 0

        self._set_status(f"Conectando a cámara ({source})...")
        self._cleanup()
        time.sleep(0.2)
        self._start_webcam(source=source)

    def _build_report_widgets(self) -> None:
        p = self._report_inner

        tk.Label(p, text="RENDIMIENTO", bg=COLORS["panel"], fg=COLORS["accent"],
                 font=FONTS["small"]).pack(fill="x", padx=12, pady=(12, 2))
        self._fps_value = self._make_stat_bar(p, "FPS", "0")
        self._runtime_value = self._make_stat_bar(p, "Backend", "-")
        self._frame_count_value = self._make_stat_bar(p, "Frames", "0")
        self._time_elapsed_value = self._make_stat_bar(p, "Tiempo", "00:00")

        sep = tk.Frame(p, bg=COLORS["border"], height=1)
        sep.pack(fill="x", padx=12, pady=8)

        tk.Label(p, text="DETECCIONES", bg=COLORS["panel"], fg=COLORS["accent"],
                 font=FONTS["small"]).pack(fill="x", padx=12, pady=(0, 2))
        self._persons_value = self._make_stat_bar(p, "Personas", "0")
        self._unprotected_value = self._make_stat_bar(p, "Sin Casco", "0")
        self._classes_frame = tk.Frame(p, bg=COLORS["panel"])
        self._classes_frame.pack(fill="x", padx=12, pady=4)

        sep2 = tk.Frame(p, bg=COLORS["border"], height=1)
        sep2.pack(fill="x", padx=12, pady=8)

        tk.Label(p, text="ALERTAS", bg=COLORS["panel"], fg=COLORS["accent"],
                 font=FONTS["small"]).pack(fill="x", padx=12, pady=(0, 2))
        self._alerts_frame = tk.Frame(p, bg=COLORS["panel"])
        self._alerts_frame.pack(fill="x", padx=12, pady=4)

    def _make_stat_bar(self, parent: tk.Widget, label: str, value: str) -> tk.Label:
        row = tk.Frame(parent, bg=COLORS["panel"])
        row.pack(fill="x", padx=12, pady=2)
        tk.Label(row, text=label, bg=COLORS["panel"], fg=COLORS["text_dim"],
                 font=FONTS["body"], anchor="w", width=12).pack(side="left")
        val = tk.Label(row, text=value, bg=COLORS["panel"], fg=COLORS["text"],
                       font=FONTS["mono"], anchor="e")
        val.pack(side="right")
        return val

    def _start_webcam(self, source: int | str = 0) -> None:
        # Loader no bloqueante si aún no cargado
        if self.app._engine is None and self.app._engine_loading:
            self._set_status("⏳ Cargando modelo YOLO, espera...")
            self.after(500, lambda: self._start_webcam(source))
            return
        self._engine = self._load_engine()
        if self._engine is None:
            self._set_status(f"✗ Error cargando modelo: {self.app._engine_error or 'desconocido'}")
            return

        self._current_source = source
        # Backend óptimo: V4L2 local, FFMPEG para RTSP/HTTP (TCP para WiFi/WAN)
        is_network = isinstance(source, str) and source.startswith(("rtsp://","rtsps://","http://","https://"))
        if is_network:
            import os
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp;stimeout;5000000"
            backend = cv2.CAP_FFMPEG
        else:
            backend = cv2.CAP_V4L2 if isinstance(source, int) and hasattr(cv2, "CAP_V4L2") else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(source, backend)
        if is_network:
            self._cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            self._cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self._cap.isOpened():
            # Inline status + dialog no bloqueante
            safe_src = str(source)[:80].replace("pass@", "***@") if isinstance(source,str) else str(source)
            self._set_status(f"✗ No se pudo abrir: {safe_src}")
            self.after(100, lambda: messagebox.showerror(
                "Error de Cámara",
                f"No se pudo abrir: '{safe_src}'.\nVerifica IP, puerto 554, user/pass y firewall. Para Dahua prueba subtype=1 (WiFi)."
            ))
            return

        if isinstance(source, int):
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)

            if source == 0:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            else:
                self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            for _ in range(5):
                self._cap.read()
        elif is_network:
            # Warmup red
            for _ in range(3):
                self._cap.read()

        self._report_data["start_time"] = time.time()
        self._report_data["frames"] = 0
        self._report_data["runtime"] = self._engine.runtime_label
        self._running = True
        self._set_status(f"Cámara activa ({source}) — {self._engine.runtime_label}")

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        self._update_display()

    def _capture_loop(self) -> None:
        while self._running and self._cap is not None:
            ok, frame = self._cap.read()
            if not ok:
                continue

            self._frame_count_idx += 1

            if self._mirror_var.get():
                frame = cv2.flip(frame, 1)

            try:
                if self._engine is None:
                    continue

                skip_interval = 2 if self._turbo_var.get() else 1
                # Si sliders custom, usar thresholds del usuario; sino usar defaults del engine
                use_custom = (abs(self._webcam_conf_var.get()-0.25) > 0.01 or
                              abs(self._webcam_ppe_var.get()-0.25) > 0.01 or
                              abs(self._webcam_helmet_var.get()-0.15) > 0.01)
                if use_custom:
                    if self._frame_count_idx % skip_interval == 0 or self._engine._last_yolo_result is None:
                        result, _ = self._engine.detect_frame(
                            frame, class_ids=self._engine.webcam_ids,
                            conf_thresh=float(self._webcam_conf_var.get()),
                            ppe_conf_thresh=float(self._webcam_ppe_var.get()),
                            helmet_conf_thresh=float(self._webcam_helmet_var.get()),
                            imgsz=416)
                        self._engine._last_yolo_result = self._engine._last_yolo_result  # ya actualizado dentro
                        self._engine._last_webcam_res = result
                    else:
                        # Reutilizar con cajas visibles (llama a engine que ya dibuja stale + cache label)
                        result = self._engine.detect_webcam_frame_skipped(
                            frame, self._frame_count_idx, skip_interval=skip_interval)
                        # Si custom, re-filtrar con thresholds custom sobre stale result
                        # (simplificado: mantener resultado stale, umbrales custom se aplican en próximo frame real)
                else:
                    result = self._engine.detect_webcam_frame_skipped(
                        frame, self._frame_count_idx, skip_interval=skip_interval
                    )
            except (RuntimeError, cv2.error) as exc:
                now = time.time()
                if now - self._last_error_report_time > 2.0:
                    self._last_error_report_time = now
                    self._run_on_ui(lambda message=str(exc): self._set_status(
                        f"Error de inferencia: {message[:120]}"
                    ))
                continue

            data = {
                "annotated": result.annotated,
                "detections": result.names,
                "alerts": result.alerts,
                "unprotected_heads": result.unprotected_heads,
                "persons": result.persons,
            }

            try:
                self._frame_queue.put_nowait(data)
            except queue.Full:
                pass

    def _update_display(self) -> None:
        if not self._running or not self.winfo_exists():
            return

        latest_data = None
        while True:
            try:
                latest_data = self._frame_queue.get_nowait()
            except queue.Empty:
                break

        if latest_data is None:
            if self.winfo_exists():
                self.after(10, self._update_display)
            return

        data = latest_data

        now = time.time()
        elapsed = now - self._report_data["start_time"]
        self._report_data["frames"] += 1
        fps = self._report_data["frames"] / max(elapsed, 0.01)
        self._report_data["fps"] = fps
        self._report_data["detections"] = data["detections"]
        self._report_data["alerts"] = data["alerts"]
        self._report_data["unprotected_heads"] = data["unprotected_heads"]
        self._report_data["persons"] = data["persons"]

        annotated = data["annotated"]
        self._update_video(annotated)
        self._update_report()
        self._set_status(f"Cámara activa — {fps:.1f} FPS")

        if self.winfo_exists():
            self.after(10, self._update_display)

    def _update_video(self, frame: np.ndarray) -> None:
        self._photo = photo_from_frame(self._canvas, frame, (854, 480))
        self._canvas.configure(image=self._photo)

    def _update_report(self) -> None:
        fps = self._report_data["fps"]
        mins = int((time.time() - self._report_data["start_time"]) / 60)
        secs = int(time.time() - self._report_data["start_time"]) % 60

        uh_color = COLORS["danger"] if self._report_data["unprotected_heads"] > 0 else COLORS["success"]
        self._fps_value.configure(text=f"{fps:.1f}")
        self._runtime_value.configure(text=self._report_data["runtime"] or "-")
        self._frame_count_value.configure(text=str(self._report_data["frames"]))
        self._time_elapsed_value.configure(text=f"{mins:02d}:{secs:02d}")
        self._persons_value.configure(text=str(self._report_data["persons"]))
        self._unprotected_value.configure(
            text=str(self._report_data["unprotected_heads"]),
            fg=uh_color,
        )

        for w in self._classes_frame.winfo_children():
            w.destroy()
        unique = sorted(set(self._report_data["detections"]))
        label_map = {"Person": "👤", "Head": "🧑", "Hands": "✋", "Gloves": "🧤",
                     "Helmet": "⛑️", "Safety-vest": "🦺", "Shoes": "👟", "Glasses": "👓"}
        for item in unique[:10]:
            icon = label_map.get(item, "•")
            n = len([x for x in self._report_data["detections"] if x == item])
            tk.Label(self._classes_frame, text=f"{icon} {item} ({n})",
                     bg=COLORS["panel"], fg=COLORS["text"],
                     font=FONTS["small"], anchor="w").pack(fill="x")

        for w in self._alerts_frame.winfo_children():
            w.destroy()
        if self._report_data["alerts"]:
            for alert in self._report_data["alerts"][:5]:
                tk.Label(self._alerts_frame, text=f"⚠ {alert}",
                         bg=COLORS["panel"], fg=COLORS["danger"],
                         font=FONTS["small"], anchor="w",
                         wraplength=300).pack(fill="x", pady=1)
        else:
            tk.Label(self._alerts_frame, text="✅ Sin alertas",
                     bg=COLORS["panel"], fg=COLORS["success"],
                     font=FONTS["small"], anchor="w").pack(fill="x")

    def _cleanup(self) -> None:
        self._running = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._thread = None
        cv2.destroyAllWindows()


class VideoView(BaseView):
    def __init__(self, parent: tk.Widget, app: MainApp) -> None:
        super().__init__(parent, app)
        self._engine: DetectionEngine | None = None
        self._processing = False
        self._stop_requested = False
        self._photo: ImageTk.PhotoImage | None = None
        self._save_path = Path("outputs/video_resultado.mp4")

        left, right = two_column_body(self)

        panel_label(left, "SELECCIONAR VIDEO")

        info = tk.Frame(left, bg=COLORS["panel"])
        info.pack(pady=6)
        self._file_label = tk.Label(info, text="Ningún archivo seleccionado",
                                    bg=COLORS["panel"], fg=COLORS["text_dim"],
                                    font=FONTS["body"], wraplength=420)
        self._file_label.pack()

        btn_row = tk.Frame(left, bg=COLORS["panel"])
        btn_row.pack(pady=4)
        tk.Button(btn_row, text="📂 Video", bg=COLORS["accent"], fg=COLORS["text"],
                  font=FONTS["big_button"], bd=0, cursor="hand2", command=self._select_file
                  ).pack(side="left", padx=6)
        tk.Button(btn_row, text="💾 Guardar como...", bg=COLORS["panel"], fg=COLORS["text"],
                  font=FONTS["small"], bd=1, cursor="hand2", command=self._select_save
                  ).pack(side="left", padx=6)
        self._save_label = tk.Label(left, text=f"Salida: {self._save_path}", bg=COLORS["panel"],
                                    fg=COLORS["text_muted"], font=FONTS["small_dim"], wraplength=420)
        self._save_label.pack(pady=2)

        # Sliders confianza video
        sl_frame = tk.Frame(left, bg=COLORS["panel"])
        sl_frame.pack(fill="x", padx=12, pady=6)
        tk.Label(sl_frame, text="Umbrales:", bg=COLORS["panel"], fg=COLORS["text_dim"], font=FONTS["small"]).pack(anchor="w")
        self._vid_conf = tk.DoubleVar(value=0.25)
        self._vid_ppe = tk.DoubleVar(value=0.25)
        self._vid_helmet = tk.DoubleVar(value=0.15)
        for txt, var, mx in [("Gral", self._vid_conf, 0.5), ("EPP", self._vid_ppe, 0.5), ("Casco", self._vid_helmet, 0.4)]:
            row = tk.Frame(sl_frame, bg=COLORS["panel"])
            row.pack(fill="x", pady=1)
            tk.Label(row, text=txt, bg=COLORS["panel"], fg=COLORS["text_muted"], font=FONTS["small_dim"], width=6, anchor="w").pack(side="left")
            tk.Scale(row, variable=var, from_=0.05, to=mx, resolution=0.05, orient="horizontal",
                     bg=COLORS["panel"], fg=COLORS["text_dim"], highlightthickness=0, length=260,
                     troughcolor=COLORS["border"], activebackground=COLORS["accent"], font=FONTS["small_dim"]
                     ).pack(side="left", fill="x", expand=True)
            tk.Label(row, textvariable=var, bg=COLORS["panel"], fg=COLORS["text_dim"], font=FONTS["mono"], width=4).pack(side="left")

        self._process_btn = action_button(
            left, "▶ Procesar", "success", self._toggle_process, state="disabled"
        )

        # Preview embebido Tk (reemplaza cv2.imshow)
        self._video_preview = tk.Label(left, bg="#000000", text="Preview del video", fg=COLORS["text_dim"], font=FONTS["small"])
        self._video_preview.pack(fill="both", expand=True, padx=8, pady=8)

        panel_label(right, "RESULTADOS")

        self._video_status = tk.Label(right, text="Esperando video...",
                                      bg=COLORS["panel"], fg=COLORS["text_dim"],
                                      font=FONTS["body"])
        self._video_status.pack(pady=5)

        self._video_result = result_text(
            right, height=12, initial="Selecciona un video y presiona Iniciar.\n"
        )

    def _select_save(self) -> None:
        p = filedialog.asksaveasfilename(title="Guardar video como", defaultextension=".mp4",
                                         filetypes=[("MP4","*.mp4"),("Todos","*.*")],
                                         initialfile=str(self._save_path.name))
        if p:
            self._save_path = Path(p)
            self._save_label.configure(text=f"Salida: {self._save_path}")

    def _select_file(self) -> None:
        path = open_system_file_dialog(
            "Seleccionar video",
            ["*.mp4", "*.avi", "*.mov", "*.mkv"],
        )
        if not path:
            return
        self._file_label.configure(text=Path(path).name, fg=COLORS["text"])
        self._file_path = path
        self._process_btn.configure(state="normal", text="▶ Iniciar Procesamiento")
        # Preview primer frame
        try:
            cap = cv2.VideoCapture(path)
            ok, fr = cap.read()
            cap.release()
            if ok:
                fr_small = cv2.resize(fr, (426,240))
                self._photo = photo_from_frame(self._video_preview, fr_small, (426,240))
                self._video_preview.configure(image=self._photo, text="")
        except Exception:
            pass

    def _toggle_process(self) -> None:
        if not self._processing:
            self._start_processing()
        else:
            self._stop_requested = True
            self._process_btn.configure(state="disabled", text="⏹ Deteniendo...")

    def _start_processing(self) -> None:
        if not hasattr(self, "_file_path"):
            return

        self._engine = self._load_engine()
        if self._engine is None:
            return

        self._processing = True
        self._stop_requested = False
        self._process_btn.configure(text="⏹ Detener", state="normal",
                                    bg=COLORS["danger"])
        replace_text(self._video_result, "Procesando video...\n")

        threading.Thread(target=self._process_video, daemon=True).start()

    def _process_video(self) -> None:
        cap = cv2.VideoCapture(self._file_path)
        if not cap.isOpened():
            self._run_on_ui(lambda: messagebox.showerror("Error", "No se pudo abrir el video"))
            self._reset_ui()
            return

        original_fps = cap.get(cv2.CAP_PROP_FPS) or 20
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        display_w, display_h = 854, 480
        out_path = self._save_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path),
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 original_fps, (display_w, display_h))
        if not writer.isOpened():
            self._run_on_ui(lambda: messagebox.showerror("Error", f"No se pudo crear: {out_path}"))
            cap.release()
            self._reset_ui()
            return

        prev_time = time.time()
        frame_count = 0
        all_detections: list[str] = []
        total_unprotected = 0

        # Preview embebido: inferencia sobre frame redimensionado directo (stretch) restaura detección previa;
        # letterbox solo afectaría filtros geométricos (área relativa) y reducía recall
        try:
            while cap.isOpened() and not self._stop_requested:
                ok, frame = cap.read()
                if not ok:
                    break

                frame_count += 1
                display_frame = cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_LINEAR)

                if self._engine is None:
                    break

                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now

                result = self._engine.detect_video_frame(
                    display_frame, fps,
                    conf_thresh=float(self._vid_conf.get()),
                    ppe_conf_thresh=float(self._vid_ppe.get()),
                    helmet_conf_thresh=float(self._vid_helmet.get()))

                writer.write(result.annotated)
                # Enviar preview a Tk thread
                preview = result.annotated.copy()
                self._run_on_ui(lambda img=preview: self._update_video_preview(img))

                all_detections.extend(result.names)
                total_unprotected += result.unprotected_heads

                progress = int(frame_count / total_frames * 100) if total_frames else 0
                self._run_on_ui(lambda p=progress, fc=frame_count, tc=total_frames,
                                current_fps=fps, current_alerts=result.alerts: (
                    replace_text(self._video_result,
                        f"Progreso: {p}% ({fc}/{tc} frames)\n"
                        f"FPS: {current_fps:.1f}\n"
                        f"Alertas: {', '.join(current_alerts) if current_alerts else 'Ninguna'}\n")
                ))
        finally:
            cap.release()
            writer.release()

        unique = sorted(set(all_detections))
        summary = (
            f"\n{'='*40}\n"
            f"VIDEO PROCESADO\n"
            f"{'='*40}\n"
            f"Frames procesados: {frame_count}\n"
            f"Clases detectadas: {', '.join(unique) if unique else 'Ninguna'}\n"
            f"Cabezas sin casco: {total_unprotected}\n"
            f"Guardado en: {out_path}\n"
        )
        self._run_on_ui(lambda: (
            replace_text(self._video_result, summary),
            self._set_status("Video procesado correctamente"),
        ))
        self._reset_ui()

    def _update_video_preview(self, frame: np.ndarray) -> None:
        try:
            if not self.winfo_exists():
                return
            self._photo = photo_from_frame(self._video_preview, frame, (854, 480))
            self._video_preview.configure(image=self._photo, text="")
        except tk.TclError:
            pass

    def _reset_ui(self) -> None:
        self._processing = False
        self._stop_requested = False
        self._run_on_ui(lambda: self._process_btn.configure(
            text="▶ Iniciar Procesamiento", state="normal", bg=COLORS["success"]))

    def _cleanup(self) -> None:
        self._stop_requested = True
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


class ImageView(BaseView):
    def __init__(self, parent: tk.Widget, app: MainApp) -> None:
        super().__init__(parent, app)
        self._engine: DetectionEngine | None = None
        self._photo: ImageTk.PhotoImage | None = None

        left, right = two_column_body(self)

        panel_label(left, "SELECCIONAR IMAGEN")

        self._img_label = tk.Label(left, text="Ninguna imagen seleccionada",
                                   bg=COLORS["panel"], fg=COLORS["text_dim"],
                                   font=FONTS["body"], wraplength=420)
        self._img_label.pack(pady=5)

        action_button(left, "📂 Seleccionar Imagen", "accent", self._select_image)

        # Sliders confianza imagen
        sl = tk.Frame(left, bg=COLORS["panel"])
        sl.pack(fill="x", padx=12, pady=4)
        tk.Label(sl, text="Umbrales:", bg=COLORS["panel"], fg=COLORS["text_dim"], font=FONTS["small"]).pack(anchor="w")
        self._img_conf = tk.DoubleVar(value=0.25)
        self._img_ppe = tk.DoubleVar(value=0.25)
        self._img_helmet = tk.DoubleVar(value=0.15)
        for txt, var, mx in [("Gral", self._img_conf, 0.5), ("EPP", self._img_ppe, 0.5), ("Casco", self._img_helmet, 0.4)]:
            row = tk.Frame(sl, bg=COLORS["panel"])
            row.pack(fill="x")
            tk.Label(row, text=txt, bg=COLORS["panel"], fg=COLORS["text_muted"], font=FONTS["small_dim"], width=6, anchor="w").pack(side="left")
            tk.Scale(row, variable=var, from_=0.05, to=mx, resolution=0.05, orient="horizontal",
                     bg=COLORS["panel"], highlightthickness=0, length=200, troughcolor=COLORS["border"],
                     activebackground=COLORS["accent"], font=FONTS["small_dim"]).pack(side="left", fill="x", expand=True)
            tk.Label(row, textvariable=var, bg=COLORS["panel"], fg=COLORS["text_dim"], font=FONTS["mono"], width=4).pack(side="left")

        self._canvas = tk.Label(left, bg="#000000", height=400, text="Preview", fg=COLORS["text_muted"])
        self._canvas.pack(fill="both", expand=True, padx=12, pady=8)

        panel_label(right, "RESULTADOS")

        self._image_result = result_text(right, height=20)

    def _select_image(self) -> None:
        path = open_system_file_dialog(
            "Seleccionar imagen",
            ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"],
        )
        if not path:
            return
        self._img_label.configure(text=Path(path).name, fg=COLORS["text"])
        self._process_image(path)

    def _process_image(self, path: str) -> None:
        if self.app._engine is None and self.app._engine_loading:
            self._set_status("⏳ Modelo cargando, espera...")
            self.after(600, lambda: self._process_image(path))
            return
        self._engine = self._load_engine()
        if self._engine is None:
            return

        self._set_status("Procesando imagen...")
        frame = cv2.imread(path)
        if frame is None:
            self._set_status("✗ No se pudo leer imagen")
            messagebox.showerror("Error", "No se pudo leer la imagen")
            return

        # Thresholds desde sliders
        result, _ = self._engine.detect_frame(
            frame, class_ids=self._engine.video_ids,
            conf_thresh=float(self._img_conf.get()), ppe_conf_thresh=float(self._img_ppe.get()),
            helmet_conf_thresh=float(self._img_helmet.get()), imgsz=640)
        from src.epp_utils import draw_status_panel as _dsp
        _dsp(result.annotated, 0.0, result.names, result.alerts)

        out = Path("outputs/imagen_resultado.jpg")
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), result.annotated)

        self._show_image(result.annotated)

        replace_text(
            self._image_result,
            f"ARCHIVO: {Path(path).name}\n"
            f"{'='*36}\n"
            f"Clases detectadas:\n"
            + "\n".join(f"  • {n}" for n in sorted(set(result.names))) +
            f"\n\nCabezas sin casco: {result.unprotected_heads}\n"
            f"Alertas: {', '.join(result.alerts) if result.alerts else 'Ninguna'}\n"
            f"Guardado en: {out}\n"
        )
        self._set_status("Imagen procesada")

    def _show_image(self, frame: np.ndarray) -> None:
        self._photo = photo_from_frame(self._canvas, frame, (600, 400))
        self._canvas.configure(image=self._photo)


class BatchView(BaseView):
    def __init__(self, parent: tk.Widget, app: MainApp) -> None:
        super().__init__(parent, app)
        self._engine: DetectionEngine | None = None
        self._processing = False

        left, right = two_column_body(self, right_width=400, right_fill="both", right_expand=True)

        panel_label(left, "PROCESAR LOTE DE IMÁGENES")

        self._folder_label = tk.Label(left, text="Ninguna carpeta seleccionada",
                                      bg=COLORS["panel"], fg=COLORS["text_dim"],
                                      font=FONTS["body"])
        self._folder_label.pack(pady=5)

        action_button(left, "📂 Seleccionar Carpeta", "accent", self._select_folder)

        self._progress = ttk.Progressbar(left, mode="determinate", length=500)
        self._progress.pack(pady=10)

        self._start_btn = action_button(
            left, "▶ Iniciar Procesamiento", "success", self._start_batch,
            state="disabled",
        )

        panel_label(right, "REGISTRO")

        self._batch_log = result_text(
            right, height=20, initial="Selecciona una carpeta y presiona Iniciar.\n"
        )

        scroll = tk.Scrollbar(right, orient="vertical", command=self._batch_log.yview)
        scroll.pack(side="right", fill="y", pady=12, ipadx=2)
        self._batch_log.configure(yscrollcommand=scroll.set)

    def _select_folder(self) -> None:
        path = open_system_directory_dialog("Seleccionar carpeta de imágenes")
        if not path:
            return
        self._folder_label.configure(text=Path(path).name, fg=COLORS["text"])
        self._input_folder = path
        self._start_btn.configure(state="normal")

    def _start_batch(self) -> None:
        if not hasattr(self, "_input_folder"):
            return

        self._engine = self._load_engine()
        if self._engine is None:
            return

        self._processing = True
        self._start_btn.configure(state="disabled", text="⏳ Procesando...")
        replace_text(self._batch_log, "")
        self._set_status("Procesando lote...")

        threading.Thread(target=self._process_batch, daemon=True).start()

    def _process_batch(self) -> None:
        input_dir = Path(self._input_folder)
        # Limitar recorrido: no seguir symlinks, filtrar archivos reales, limitar cantidad
        MAX_BATCH_IMAGES = 1000
        images = sorted(
            p for p in input_dir.rglob("*")
            if p.is_file() and not p.is_symlink() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(images) > MAX_BATCH_IMAGES:
            self._run_on_ui(lambda: messagebox.showwarning(
                "Lote grande",
                f"Se encontraron {len(images)} imágenes. Se procesarán solo las primeras {MAX_BATCH_IMAGES} por seguridad."
            ))
            images = images[:MAX_BATCH_IMAGES]
        total = len(images)
        output_dir = Path("outputs/batch")
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "resultados_batch.csv"

        def log(msg: str) -> None:
            self._run_on_ui(lambda text=msg: (
                self._batch_log.insert("end", text + "\n"),
                self._batch_log.see("end"),
            ))

        log(f"Imágenes encontradas: {total}")
        log("Procesando...\n")

        processed = 0

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "detected_classes", "alerts", "output_image"])

            for i, img_path in enumerate(images):
                if not self._processing:
                    break

                frame = cv2.imread(str(img_path))
                if frame is None:
                    log(f"✗ Omitido: {img_path.name}")
                    continue

                if self._engine is None:
                    break

                result = self._engine.detect_batch_frame(frame)

                out_path = output_dir / f"{img_path.stem}_detected.jpg"
                cv2.imwrite(str(out_path), result.annotated)

                # Sanitizar CSV contra formula injection (prefijar con ' si empieza con =,+,-,@)
                def csv_safe(s: str) -> str:
                    return "'" + s if s and s[0] in "=@+-" else s
                writer.writerow([
                    csv_safe(str(img_path)),
                    csv_safe(";".join(sorted(set(result.names)))),
                    csv_safe(";".join(result.alerts)),
                    csv_safe(str(out_path)),
                ])

                processed += 1

                pct = int((i + 1) / total * 100) if total else 0
                self._run_on_ui(lambda p=pct: self._progress.configure(value=p))
                log(f"✓ {img_path.name} — {', '.join(result.alerts) if result.alerts else 'OK'}")

        summary = (
            f"\n{'='*36}\n"
            f"LOTE COMPLETADO\n"
            f"{'='*36}\n"
            f"Procesadas: {processed}/{total}\n"
            f"CSV: {csv_path}\n"
        )
        log(summary)
        self._run_on_ui(lambda: (
            self._set_status(f"Lote completado: {processed}/{total} imágenes"),
            self._start_btn.configure(state="normal", text="▶ Iniciar Procesamiento"),
        ))
        self._processing = False

    def _cleanup(self) -> None:
        self._processing = False
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
