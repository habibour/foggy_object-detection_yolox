"""
prepare_voc.py — Convert clean VOC (Pascal VOC 2007+2012) to YOLO format
==========================================================================
Filters VOC's 20 classes down to the 5 RTTS-matching classes:
  person, bicycle, car, bus, motorbike

Usage:
    python prepare_voc.py --voc-dir /path/to/VOCdevkit --out-dir /path/to/VOC_YOLO

Input:  Pascal VOC format (XML annotations)
Output: YOLO format (images/ + labels/ with train/val splits)
"""

import os
import sys
import shutil
import random
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

# ── 5 classes matching RTTS ──────────────────────────────────────────────────
RTTS_CLASSES = ['person', 'bicycle', 'car', 'bus', 'motorbike']
# VOC class name → RTTS class index
CLASS_MAP = {name: idx for idx, name in enumerate(RTTS_CLASSES)}

# VOC has different naming for some:
VOC_TO_RTTS = {
    'person':    'person',
    'bicycle':   'bicycle',
    'car':       'car',
    'bus':       'bus',
    'motorbike': 'motorbike',
}


def parse_voc_xml(xml_path: str) -> list:
    """Parse VOC XML annotation, return list of (class_name, bbox) for RTTS classes only."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    if w == 0 or h == 0:
        return [], 0, 0

    objects = []
    for obj in root.findall('object'):
        cls_name = obj.find('name').text.strip()

        # Only keep RTTS classes
        if cls_name not in VOC_TO_RTTS:
            continue

        difficult = obj.find('difficult')
        if difficult is not None and int(difficult.text) == 1:
            continue  # skip difficult samples

        rtts_name = VOC_TO_RTTS[cls_name]
        cls_idx = CLASS_MAP[rtts_name]

        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        # Convert to YOLO format: x_center, y_center, width, height (normalized)
        x_center = ((xmin + xmax) / 2) / w
        y_center = ((ymin + ymax) / 2) / h
        box_w = (xmax - xmin) / w
        box_h = (ymax - ymin) / h

        # Clamp
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        box_w = max(0, min(1, box_w))
        box_h = max(0, min(1, box_h))

        objects.append((cls_idx, x_center, y_center, box_w, box_h))

    return objects, w, h


def convert_voc_to_yolo(voc_dir: str, out_dir: str, val_ratio: float = 0.15,
                         test_ratio: float = 0.15, seed: int = 42):
    """
    Convert VOC dataset to YOLO format, keeping only RTTS 5 classes.

    Args:
        voc_dir:   Path to VOCdevkit or directory containing Annotations/ and JPEGImages/
        out_dir:   Output directory (will create images/ and labels/ subdirs)
        val_ratio: Fraction of data for validation
        seed:      Random seed for reproducibility
    """
    voc_dir = Path(voc_dir)
    out_dir = Path(out_dir)

    # Try common VOC directory structures
    ann_dirs = []
    img_dirs = []
    for subdir in ['VOC2007', 'VOC2012', '.']:
        ann = voc_dir / subdir / 'Annotations'
        img = voc_dir / subdir / 'JPEGImages'
        if ann.exists() and img.exists():
            ann_dirs.append(ann)
            img_dirs.append(img)

    if not ann_dirs:
        # Try direct path
        if (voc_dir / 'Annotations').exists():
            ann_dirs = [voc_dir / 'Annotations']
            img_dirs = [voc_dir / 'JPEGImages']
        else:
            raise FileNotFoundError(
                f"Could not find Annotations/ and JPEGImages/ in {voc_dir}\n"
                f"Tried: {voc_dir}/VOC2007, {voc_dir}/VOC2012, {voc_dir}/"
            )

    print(f"Found {len(ann_dirs)} VOC annotation directories:")
    for d in ann_dirs:
        print(f"  {d}")

    # Create output dirs
    for split in ['train', 'val', 'test']:
        (out_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Process all annotations
    valid_samples = []  # (xml_path, img_path, objects)
    skipped = 0

    for ann_dir, img_dir in zip(ann_dirs, img_dirs):
        for xml_file in sorted(ann_dir.glob('*.xml')):
            objects, w, h = parse_voc_xml(str(xml_file))

            if not objects:  # no RTTS classes in this image
                skipped += 1
                continue

            # Find corresponding image
            stem = xml_file.stem
            img_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG']:
                candidate = img_dir / (stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break

            if img_path is None:
                skipped += 1
                continue

            valid_samples.append((xml_file, img_path, objects))

    print(f"\nTotal images with RTTS classes: {len(valid_samples)}")
    print(f"Skipped (no matching classes):  {skipped}")

    # Split train/val/test
    random.seed(seed)
    random.shuffle(valid_samples)
    n_val = int(len(valid_samples) * val_ratio)
    n_test = int(len(valid_samples) * test_ratio)
    val_samples = valid_samples[:n_val]
    test_samples = valid_samples[n_val:n_val + n_test]
    train_samples = valid_samples[n_val + n_test:]

    print(f"Train: {len(train_samples)}")
    print(f"Val:   {len(val_samples)}")
    print(f"Test:  {len(test_samples)}")

    # Write files
    class_counts = {name: 0 for name in RTTS_CLASSES}

    for split, samples in [('train', train_samples), ('val', val_samples), ('test', test_samples)]:
        for xml_path, img_path, objects in samples:
            stem = img_path.stem
            dst_img = out_dir / 'images' / split / img_path.name
            dst_lbl = out_dir / 'labels' / split / (stem + '.txt')

            # Copy image
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)

            # Write YOLO label
            with open(dst_lbl, 'w') as f:
                for cls_idx, xc, yc, bw, bh in objects:
                    f.write(f"{cls_idx} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                    class_counts[RTTS_CLASSES[cls_idx]] += 1

    # Print class distribution
    print(f"\nClass distribution (all splits):")
    for name, count in class_counts.items():
        print(f"  {name:12s}: {count:6d}")

    # Write YAML config
    yaml_path = out_dir / 'voc.yaml'
    yaml_content = f"""# Clean VOC Dataset — 5 RTTS classes
# Auto-generated by prepare_voc.py

path: {out_dir}
train: images/train
val: images/val
test: images/test

nc: 5
names:
  0: person
  1: bicycle
  2: car
  3: bus
  4: motorbike
"""
    yaml_path.write_text(yaml_content)
    print(f"\nYAML config: {yaml_path}")
    print(f"Done! {len(valid_samples)} images converted.")
    return str(yaml_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert VOC to YOLO (5 RTTS classes)')
    parser.add_argument('--voc-dir', type=str, required=True,
                        help='Path to VOCdevkit or VOC directory')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Output directory (default: VOC_YOLO/ next to voc-dir)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='Validation split ratio (default: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                        help='Test split ratio (default: 0.15)')
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = str(Path(args.voc_dir).parent / 'VOC_YOLO')

    convert_voc_to_yolo(args.voc_dir, args.out_dir, args.val_ratio, args.test_ratio)
