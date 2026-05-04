# PCB Automated Optical Inspection (AOI)

Real-time PCB solder defect detection using YOLOv8 + ONNX.  
Trained on the [PCB-AoI industrial dataset](https://www.kaggle.com/datasets/kubeedgeianvs/pcb-aoi) â€” real SPI/AOI thermal imagery.

![Demo Results](assets/demo_results.png)

## Results

| Metric | Value |
|--------|-------|
| mAP@50 | **90.4%** |
| Inference Latency | **14ms** |
| Throughput | **71.6 FPS** |
| Bad_podu mAP | 92.6% |
| Bad_qiaojiao mAP | 88.2% |

## Business Value

- Inspects **71 boards/second** â€” no human inspector matches this
- Catches **87 of 100 defects** before they leave the line
- **90% precision** â€” 1 in 10 alerts is a false alarm
- **Edge deployment** â€” runs on-site, no cloud, no subscription

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/pcb-aoi-inspection
cd pcb-aoi-inspection
pip install -r requirements.txt

# Run inference
python inference/inference_onnx.py \
    --model models/pcb_defect.onnx \
    --source path/to/pcb_image.jpg
```

## Dataset

Download from Kaggle and convert:
```bash
python data/convert_voc_to_yolo.py
```

## Training

```bash
python train/train.py
```

## Tech Stack

- **Model**: YOLOv8s (11M parameters)
- **Format**: ONNX â€” runs without PyTorch
- **Dataset**: PCB-AoI (SPI/AOI thermal imaging)
- **Defects**: Bad_podu آ· Bad_qiaojiao

## License
MIT
