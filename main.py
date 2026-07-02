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
from ultralytics import YOLO

from src.epp_utils import (
    DEFAULT_IOU,
    IMAGE_EXTENSIONS,
    SAFETY_VEST_CLASS_ID,
    VIDEO_DISPLAY_CLASS_IDS,
    WEBCAM_CLASS_IDS,
    analyze_compliance,
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
    "text_dim": "#8b949e",
    "border": "#30363d",
}

FONTS = {
    "title": ("Segoe UI", 22, "bold"),
    "heading": ("Segoe UI", 14, "bold"),
    "body": ("Segoe UI", 11),
    "mono": ("Consolas", 11),
    "small": ("Segoe UI", 9),
    "big_button": ("Segoe UI", 13, "bold"),
}


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
    cw = label.winfo_width() or fallback_size[0]
    ch = label.winfo_height() or fallback_size[1]
    h, w = frame.shape[:2]
    scale = min(cw / w, ch / h)
    resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(image=Image.fromarray(rgb))


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


@dataclass
class DetectionResult:
    annotated: np.ndarray
    names: list[str]
    unprotected_heads: int
    alerts: list[str]
    persons: int = 0


class DetectionEngine:
    def __init__(self, model_path: str = "weights/best.pt") -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo: {self.model_path}")

        self.model = YOLO(str(self.model_path))
        self.webcam_ids = filter_supported_class_ids(self.model, WEBCAM_CLASS_IDS)
        self.video_ids = filter_supported_class_ids(self.model, VIDEO_DISPLAY_CLASS_IDS)
        self.check_vest = model_supports_class_id(self.model, SAFETY_VEST_CLASS_ID)

    def _predict(self, frame: np.ndarray, conf: float, class_ids: list[int],
                 imgsz: int) -> object:
        return self.model.predict(
            frame,
            conf=conf,
            iou=DEFAULT_IOU,
            imgsz=imgsz,
            classes=class_ids,
            verbose=False,
        )[0]

    def _detect(
        self,
        frame: np.ndarray,
        class_ids: list[int],
        conf_thresh: float,
        ppe_conf_thresh: float,
        helmet_conf_thresh: float,
        imgsz: int,
    ) -> DetectionResult:
        inference_conf = min(conf_thresh, ppe_conf_thresh, helmet_conf_thresh)
        result = self._predict(frame, inference_conf, class_ids, imgsz)
        annotated, names, heads = draw_detection_boxes(
            frame,
            result,
            self.model,
            scale_x=1.0,
            scale_y=1.0,
            class_ids=class_ids,
            conf_thresh=conf_thresh,
            ppe_conf_thresh=ppe_conf_thresh,
            helmet_conf_thresh=helmet_conf_thresh,
        )
        alerts = analyze_compliance(names, heads, check_vest=self.check_vest)
        persons = sum(1 for name in names if name == "Person")
        return DetectionResult(annotated, names, int(heads), alerts, persons)

    def detect_webcam_frame(self, frame: np.ndarray) -> DetectionResult:
        return self._detect(
            frame,
            class_ids=self.webcam_ids,
            conf_thresh=0.25,
            ppe_conf_thresh=0.25,
            helmet_conf_thresh=0.15,
            imgsz=416,
        )

    def detect_image_frame(self, frame: np.ndarray) -> DetectionResult:
        result = self._detect(
            frame,
            class_ids=self.video_ids,
            conf_thresh=0.25,
            ppe_conf_thresh=0.25,
            helmet_conf_thresh=0.15,
            imgsz=640,
        )
        draw_status_panel(result.annotated, 0.0, result.names, result.alerts)
        return result

    def detect_batch_frame(self, frame: np.ndarray) -> DetectionResult:
        return self.detect_image_frame(frame)

    def detect_video_frame(self, frame: np.ndarray, fps: float) -> DetectionResult:
        result = self._detect(
            frame,
            class_ids=self.video_ids,
            conf_thresh=0.50,
            ppe_conf_thresh=0.45,
            helmet_conf_thresh=0.15,
            imgsz=640,
        )
        draw_status_panel_big(result.annotated, fps, result.names, result.alerts)
        return result


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

        self._setup_styles()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
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
            self._engine = DetectionEngine()
        return self._engine


class MainMenu(tk.Frame):
    def __init__(self, parent: tk.Widget, app: MainApp) -> None:
        super().__init__(parent, bg=COLORS["bg"])
        self.app = app

        header = tk.Frame(self, bg=COLORS["bg"])
        header.pack(fill="x", pady=(60, 10))
        tk.Label(header, text="Sistema de Detección de EPP",
                 bg=COLORS["bg"], fg=COLORS["text"], font=FONTS["title"]).pack()
        tk.Label(header, text="Equipo de Protección Personal — Construcción",
                 bg=COLORS["bg"], fg=COLORS["text_dim"], font=FONTS["body"]).pack()

        cards = tk.Frame(self, bg=COLORS["bg"])
        cards.pack(expand=True)

        options = [
            ("Cámara en Vivo", "Detección en tiempo real\ncon cámara", "🎥", "success", WebcamView),
            ("Video", "Procesar archivo de video\ncon detección", "🎬", "accent", VideoView),
            ("Imagen", "Detectar EPP en una\nimagen estática", "🖼️", "warning", ImageView),
            ("Lote de Imágenes", "Procesar carpeta\nde imágenes", "📁", "danger", BatchView),
        ]

        for start in range(0, len(options), 2):
            row = tk.Frame(cards, bg=COLORS["bg"])
            row.pack(pady=10)
            for title, desc, icon, color_key, view in options[start:start + 2]:
                CardButton(row, title, desc, icon, COLORS[color_key],
                           lambda target=view: app._show_frame(target),
                           width=260, height=200).pack(side="left", padx=12)

        tk.Label(self, text="Selecciona un modo para comenzar",
                 bg=COLORS["bg"], fg=COLORS["text_dim"], font=FONTS["body"]).pack(pady=(10, 40))


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
        self._report_data = {
            "fps": 0.0, "detections": [], "alerts": [],
            "unprotected_heads": 0, "persons": 0, "frames": 0,
            "start_time": 0.0, "has_vest_model": False,
        }

        body = tk.Frame(self, bg=COLORS["bg"])
        body.pack(fill="both", expand=True, padx=8, pady=8)

        video_frame = tk.LabelFrame(body, text="VIDEO EN VIVO", bg=COLORS["panel"],
                                    fg=COLORS["accent"], font=FONTS["heading"],
                                    highlightbackground=COLORS["border"],
                                    highlightthickness=1, bd=0)
        video_frame.pack(side="left", fill="both", expand=True)
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
        self._start_webcam()

    def _build_report_widgets(self) -> None:
        p = self._report_inner

        tk.Label(p, text="RENDIMIENTO", bg=COLORS["panel"], fg=COLORS["accent"],
                 font=FONTS["small"]).pack(fill="x", padx=12, pady=(12, 2))
        self._fps_value = self._make_stat_bar(p, "FPS", "0")
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

    def _start_webcam(self) -> None:
        self._engine = self._load_engine()
        if self._engine is None:
            self.after(0, lambda: self.app._show_frame(MainMenu))
            return

        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            messagebox.showerror("Error", "No se pudo abrir la cámara.")
            self.after(0, lambda: self.app._show_frame(MainMenu))
            return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self._report_data["start_time"] = time.time()
        self._running = True
        self._set_status("Cámara iniciada")

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        self._update_display()

    def _capture_loop(self) -> None:
        while self._running and self._cap is not None:
            ok, frame = self._cap.read()
            if not ok:
                continue

            #frame = cv2.flip(frame, 2)

            try:
                if self._engine is None:
                    continue

                result = self._engine.detect_webcam_frame(frame)
            except Exception:
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

        try:
            data = self._frame_queue.get_nowait()
        except queue.Empty:
            if self.winfo_exists():
                self.after(16, self._update_display)
            return

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
            self.after(16, self._update_display)

    def _update_video(self, frame: np.ndarray) -> None:
        self._photo = photo_from_frame(self._canvas, frame, (854, 480))
        self._canvas.configure(image=self._photo)

    def _update_report(self) -> None:
        fps = self._report_data["fps"]
        mins = int((time.time() - self._report_data["start_time"]) / 60)
        secs = int(time.time() - self._report_data["start_time"]) % 60

        uh_color = COLORS["danger"] if self._report_data["unprotected_heads"] > 0 else COLORS["success"]
        self._fps_value.configure(text=f"{fps:.1f}")
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

        left, right = two_column_body(self)

        panel_label(left, "SELECCIONAR VIDEO")

        info = tk.Frame(left, bg=COLORS["panel"])
        info.pack(pady=10)
        self._file_label = tk.Label(info, text="Ningún archivo seleccionado",
                                    bg=COLORS["panel"], fg=COLORS["text_dim"],
                                    font=FONTS["body"])
        self._file_label.pack()

        action_button(left, "📂 Seleccionar Video", "accent", self._select_file)

        self._process_btn = action_button(
            left, "▶ Procesar", "success", self._toggle_process, state="disabled"
        )

        panel_label(right, "RESULTADOS")

        self._video_status = tk.Label(right, text="Esperando video...",
                                      bg=COLORS["panel"], fg=COLORS["text_dim"],
                                      font=FONTS["body"])
        self._video_status.pack(pady=5)

        self._video_result = result_text(
            right, height=20, initial="Selecciona un video y presiona Iniciar.\n"
        )

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
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        display_w, display_h = 854, 480
        out_path = Path("outputs/video_resultado.mp4")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path),
                                 cv2.VideoWriter.fourcc("m", "p", "4", "v"),
                                 original_fps, (display_w, display_h))

        prev_time = time.time()
        frame_count = 0
        all_detections: list[str] = []
        total_unprotected = 0

        window = "EPP Construcción - Video"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, display_w, display_h)

        try:
            while cap.isOpened() and not self._stop_requested:
                ok, frame = cap.read()
                if not ok:
                    break

                frame_count += 1
                display_frame = cv2.resize(frame, (display_w, display_h))

                if self._engine is None:
                    break

                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now

                result = self._engine.detect_video_frame(display_frame, fps)

                writer.write(result.annotated)
                cv2.imshow(window, result.annotated)

                all_detections.extend(result.names)
                total_unprotected += result.unprotected_heads

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                progress = int(frame_count / total_frames * 100) if total_frames else 0
                self._run_on_ui(lambda p=progress, fc=frame_count, tc=total_frames,
                                current_fps=fps, current_alerts=result.alerts: (
                    replace_text(self._video_result,
                        f"Progreso: {p}% ({fc}/{tc} frames)\n"
                        f"FPS: {current_fps:.1f}\n"
                        f"Alertas actuales: {', '.join(current_alerts) if current_alerts else 'Ninguna'}\n")
                ))
        finally:
            cap.release()
            writer.release()
            try:
                cv2.destroyWindow(window)
            except cv2.error:
                pass

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

    def _reset_ui(self) -> None:
        self._processing = False
        self._stop_requested = False
        self._run_on_ui(lambda: self._process_btn.configure(
            text="▶ Iniciar Procesamiento", state="normal", bg=COLORS["success"]))

    def _cleanup(self) -> None:
        self._stop_requested = True
        cv2.destroyAllWindows()


class ImageView(BaseView):
    def __init__(self, parent: tk.Widget, app: MainApp) -> None:
        super().__init__(parent, app)
        self._engine: DetectionEngine | None = None
        self._photo: ImageTk.PhotoImage | None = None

        left, right = two_column_body(self)

        panel_label(left, "SELECCIONAR IMAGEN")

        self._img_label = tk.Label(left, text="Ninguna imagen seleccionada",
                                   bg=COLORS["panel"], fg=COLORS["text_dim"],
                                   font=FONTS["body"])
        self._img_label.pack(pady=5)

        action_button(left, "📂 Seleccionar Imagen", "accent", self._select_image)

        self._canvas = tk.Label(left, bg="#000000", height=400)
        self._canvas.pack(fill="both", expand=True, padx=12, pady=12)

        panel_label(right, "RESULTADOS")

        self._image_result = result_text(right, height=25)

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
        self._engine = self._load_engine()
        if self._engine is None:
            return

        self._set_status("Procesando imagen...")
        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Error", "No se pudo leer la imagen")
            return

        result = self._engine.detect_image_frame(frame)

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

        images = sorted(
            p for p in input_dir.rglob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
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

                writer.writerow([
                    str(img_path),
                    ";".join(sorted(set(result.names))),
                    ";".join(result.alerts),
                    str(out_path),
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


if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
