"""
VOC XML â†’ YOLO Format Converter
For PCB-AoI dataset from Kaggle:
https://www.kaggle.com/datasets/kubeedgeianvs/pcb-aoi
"""

import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path


def discover_classes(annotation_dirs: list) -> list:
    classes = set()
    for d in annotation_dirs:
        for xml_file in Path(d).glob("*.xml"):
            for obj in ET.parse(xml_file).getroot().findall("object"):
                classes.add(obj.find("name").text.strip())
    return sorted(classes)


def convert_voc_to_yolo(xml_path: Path, class_names: list):
    root  = ET.parse(xml_path).getroot()
    size  = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    if img_w == 0 or img_h == 0:
        return None, 0

    lines, skipped = [], 0
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in class_names:
            skipped += 1
            continue
        b    = obj.find("bndbox")
        xmin = float(b.find("xmin").text)
        ymin = float(b.find("ymin").text)
        xmax = float(b.find("xmax").text)
        ymax = float(b.find("ymax").text)

        if xmax <= xmin or ymax <= ymin:
            skipped += 1
            continue

        cx = max(0., min(1., (xmin + xmax) / 2 / img_w))
        cy = max(0., min(1., (ymin + ymax) / 2 / img_h))
        w  = max(0., min(1., (xmax - xmin) / img_w))
        h  = max(0., min(1., (ymax - ymin) / img_h))
        lines.append(f"{class_names.index(name)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines, skipped


def parse_index(index_path: Path, base_dir: Path) -> list:
    samples = []
    for line in index_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        img = base_dir / "JPEGImages"  / Path(parts[0]).name
        ann = base_dir / "Annotations" / Path(parts[1]).name
        if img.exists() and ann.exists():
            samples.append((img, ann))
    return samples


def convert_dataset(dataset_root: str, output_root: str, train_split: float = 0.85):
    dataset_root = Path(dataset_root)
    output_root  = Path(output_root)
    random.seed(42)

  
    class_names = discover_classes([
        dataset_root / "train_data"              / "Annotations",
        dataset_root / "test_data"               / "Annotations",
        dataset_root / "train_data_augmentation" / "Annotations",
    ])
    print(f"Classes ({len(class_names)}): {class_names}")

   
    all_train = (
        parse_index(dataset_root / "train_data" / "index.txt",
                    dataset_root / "train_data") +
        parse_index(dataset_root / "train_data_augmentation" / "index.txt",
                    dataset_root / "train_data_augmentation")
    )
    test_samples = parse_index(
        dataset_root / "test_data" / "index.txt",
        dataset_root / "test_data"
    )

    random.shuffle(all_train)
    split        = int(len(all_train) * train_split)
    final_splits = {
        "train": all_train[:split],
        "val":   all_train[split:] + test_samples,
    }


    for name, samples in final_splits.items():
        img_dir = output_root / "images" / name
        lbl_dir = output_root / "labels" / name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        ok = 0
        for img_path, ann_path in samples:
            lines, _ = convert_voc_to_yolo(ann_path, class_names)
            if lines is None:
                continue
            shutil.copy2(img_path, img_dir / img_path.name)
            (lbl_dir / (img_path.stem + ".txt")).write_text("\n".join(lines))
            ok += 1
        print(f"{name}: {ok} images âœ…")

    # data.yaml
    yaml = f"""path: {output_root.absolute()}
train: images/train
val:   images/val
nc: {len(class_names)}
names:\n"""
    for i, n in enumerate(class_names):
        yaml += f"  {i}: {n}\n"
    (output_root / "data.yaml").write_text(yaml)
    print(f"\nâœ… data.yaml saved")


if __name__ == "__main__":
    convert_dataset(
        dataset_root = "./data/PCB-AoI",
        output_root  = "./data/pcb_yolo",
    )