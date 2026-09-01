"""Entrenamiento mejorado YOLOv8 SH17-construcción (10 clases).
Mejoras vs. baseline:
- Balanceo por oversampling de clases minoritarias (Safety-vest, Glasses, Ear, Shoes)
- Stratified split por presencia de clases raras
- Augment intensificado para objetos pequeños (Hands/Gloves/Helmet/Glasses): mosaic 1.0, mixup 0.15, copy_paste 0.3, hsv, scale 0.5
- Hyperparams: AdamW, cos_lr, warmup, box/cls/dfl tunings, close_mosaic 15, patience 25
- Métricas por clase + PR/F1 curvas + confusion_matrix
Uso colab: python src/train.py --data /content/.../sh17_construction_data.yaml
Uso local: python src/train.py --data datasets/sh17_construction_filtered/sh17_construction_data.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Entrenamiento YOLOv8 mejorado SH17")
    p.add_argument("--data", required=True, help="Ruta a sh17_construction_data.yaml")
    p.add_argument("--model", default="yolov8s.pt", help="yolov8n/s/m.pt")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--project", default="runs")
    p.add_argument("--name", default="sh17_construction_yolov8s_v2")
    p.add_argument("--device", default=0, help="0 o cpu")
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)

    # Hyperparams optimizados para EPP pequeño y desbalanceo
    # Ver https://docs.ultralytics.com/modes/train/#train-settings
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        plots=True,
        save=True,
        save_period=-1,
        val=True,
        verbose=True,
        optimizer="AdamW",
        lr0=0.002,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.6,  # ↑ para clases minoritarias (Safety-vest/Glasses)
        dfl=1.5,
        pose=12.0,
        kobj=1.0,
        label_smoothing=0.0,
        nbs=64,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        # Augment
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0001,
        flipud=0.0,
        fliplr=0.5,
        bgr=0.0,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.3,  # clave para Hands/Gloves pequeños
        copy_paste_mode="mixup",
        auto_augment="randaugment",
        erasing=0.2,
        crop_fraction=1.0,
        # Scheduler
        cos_lr=True,
        close_mosaic=15,
        patience=args.patience,
        # Resume
        resume=args.resume,
    )
    print("✅ Entrenamiento finalizado:", results)
    # Val final con plots
    best = Path(args.project) / args.name / "weights" / "best.pt"
    if best.exists():
        m = YOLO(str(best))
        m.val(data=args.data, imgsz=args.imgsz, device=args.device, plots=True, verbose=True)
        try:
            print("ONNX export...")
            m.export(format="onnx", imgsz=args.imgsz, opset=12)
        except Exception as e:
            print("ONNX skip:", e)


if __name__ == "__main__":
    main()
