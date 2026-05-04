"""
YOLOv8s Training Script â€” PCB-AoI Defect Detection
"""

from pathlib import Path
from ultralytics import YOLO
import torch


def train(
    data_yaml : str  = "./data/pcb_yolo/data.yaml",
    model_size: str  = "s",
    epochs    : int  = 80,
    batch     : int  = 16,
    imgsz     : int  = 640,
    project   : str  = "./runs",
    name      : str  = "pcb_aoi",
):
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    model   = YOLO(f"yolov8{model_size}.pt")
    results = model.train(
        data          = data_yaml,
        epochs        = epochs,
        imgsz         = imgsz,
        batch         = batch,
        device        = device,
        optimizer     = "AdamW",
        lr0           = 0.001,
        lrf           = 0.01,
        warmup_epochs = 5,
        cls           = 2.0,
        hsv_h         = 0.0,
        hsv_s         = 0.3,
        hsv_v         = 0.4,
        fliplr        = 0.5,
        flipud        = 0.3,
        degrees       = 5.0,
        mosaic        = 0.8,
        mixup         = 0.1,
        project       = project,
        name          = name,
        plots         = True,
        save          = True,
    )

    print(f"\n{'='*40}")
    print(f"mAP@50:    {results.results_dict['metrics/mAP50(B)']:.3f}")
    print(f"Precision: {results.results_dict['metrics/precision(B)']:.3f}")
    print(f"Recall:    {results.results_dict['metrics/recall(B)']:.3f}")

    # Export ONNX
    best = Path(project) / name / "weights" / "best.pt"
    YOLO(str(best)).export(format="onnx", imgsz=imgsz, opset=12, simplify=True)
    print(f"\nâœ… ONNX exported")


if __name__ == "__main__":
    train()