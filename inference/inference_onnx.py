"""
ONNX Runtime Inference â€” PCB Defect Detection
Runs on any hardware without PyTorch dependency.

Usage:
    python inference/inference_onnx.py \
        --model models/pcb_defect.onnx \
        --source path/to/image.jpg
"""

import cv2
import numpy as np
import onnxruntime as ort
import argparse
import time
from pathlib import Path

CLASS_NAMES = ["Bad_podu", "Bad_qiaojiao"]
COLORS      = [(80, 80, 255), (80, 255, 80)]


def build_session(model_path: str) -> ort.InferenceSession:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session   = ort.InferenceSession(model_path, providers=providers)
    print(f"Provider: {session.get_providers()[0]}")
    return session


def preprocess(img_bgr: np.ndarray, imgsz: int = 640):
    h, w  = img_bgr.shape[:2]
    rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rsz   = cv2.resize(rgb, (imgsz, imgsz))
    blob  = rsz.astype(np.float32) / 255.0
    blob  = np.transpose(blob, (2, 0, 1))[None]
    return blob, h, w


def postprocess(outputs, orig_h, orig_w,
                conf_thresh=0.25, iou_thresh=0.45):
    pred     = outputs[0][0].T
    boxes_xy = pred[:, :4]
    scores   = pred[:, 4:]
    cls_ids  = np.argmax(scores, axis=1)
    confs    = scores[np.arange(len(scores)), cls_ids]

    mask     = confs >= conf_thresh
    boxes_xy = boxes_xy[mask]
    confs    = confs[mask]
    cls_ids  = cls_ids[mask]

    if len(boxes_xy) == 0:
        return []

    x1 = boxes_xy[:, 0] - boxes_xy[:, 2] / 2
    y1 = boxes_xy[:, 1] - boxes_xy[:, 3] / 2
    x2 = boxes_xy[:, 0] + boxes_xy[:, 2] / 2
    y2 = boxes_xy[:, 1] + boxes_xy[:, 3] / 2
    xyxy = np.stack([x1, y1, x2, y2], axis=1)

    indices = cv2.dnn.NMSBoxes(
        xyxy.tolist(), confs.tolist(), conf_thresh, iou_thresh
    )

    sx, sy = orig_w / 640, orig_h / 640
    results = []
    for i in indices:
        idx = int(i)
        results.append({
            "class": int(cls_ids[idx]),
            "name":  CLASS_NAMES[int(cls_ids[idx])],
            "conf":  float(confs[idx]),
            "bbox":  [int(xyxy[idx,0]*sx), int(xyxy[idx,1]*sy),
                      int(xyxy[idx,2]*sx), int(xyxy[idx,3]*sy)],
        })
    return results


def draw(img: np.ndarray, detections: list) -> np.ndarray:
    out = img.copy()
    for det in detections:
        x1,y1,x2,y2 = det["bbox"]
        color = COLORS[det["class"] % len(COLORS)]
        label = f"{det['name']} {det['conf']:.0%}"
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        cv2.putText(out, label, (x1, max(y1-6,10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return out


def run(model_path: str, source: str, save: bool = True):
    session    = build_session(model_path)
    input_name = session.get_inputs()[0].name

    img   = cv2.imread(source)
    blob, h, w = preprocess(img)

    # warmup
    for _ in range(3):
        session.run(None, {input_name: blob})

    t0      = time.perf_counter()
    outputs = session.run(None, {input_name: blob})
    ms      = (time.perf_counter() - t0) * 1000

    dets = postprocess(outputs, h, w)
    print(f"Latency: {ms:.1f}ms | Detected: {len(dets)} defects")
    for d in dets:
        print(f"  {d['name']:20s} {d['conf']:.0%}  {d['bbox']}")

    result = draw(img, dets)

    if save:
        out_path = Path(source).stem + "_result.jpg"
        cv2.imwrite(out_path, result)
        print(f"Saved: {out_path}")

    return result, dets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--save",   action="store_true", default=True)
    args = parser.parse_args()
    run(args.model, args.source, args.save)