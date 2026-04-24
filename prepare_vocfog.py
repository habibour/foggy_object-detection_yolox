"""
prepare_vocfog.py — Converts your existing voc-fog(9578+2129) folder
into YOLO format for Stage 1 pre-training of the SAID model.

Your folder structure (already exists):
  voc-fog(9578+2129)/
    train/
      Annotations/       ← VOC XML labels
      VOC2007-FOG/       ← foggy .jpg images  ← we use THESE (not JPEGImages)
    test/
      Annotations/
      VOCtest-FOG/       ← foggy .jpg images

Output YOLO structure (created at ./VOC_FOG_YOLO/):
  VOC_FOG_YOLO/
    images/
      train/  ← symlinked / copied foggy images
      val/    ← 10% of test set used as val
      test/   ← remaining 90% of test set
    labels/
      train/  ← converted YOLO .txt labels
      val/
      test/

Also writes vocfog.yaml next to this script.

Usage:
    python prepare_vocfog.py
"""

import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
VOC_FOG    = ROOT / "voc-fog(9578+2129)"
OUTPUT     = ROOT / "VOC_FOG_YOLO"
YAML_PATH  = ROOT / "vocfog.yaml"

# Only keep classes that appear in RTTS (our final target domain)
CLASSES    = ["person", "bicycle", "car", "bus", "motorbike"]

# Fraction of the test set to use as validation
VAL_RATIO  = 0.10
SEED       = 42


# ─────────────────────────────────────────────────────────────────────────────
# VOC XML → YOLO label converter
# ─────────────────────────────────────────────────────────────────────────────
def voc_to_yolo(xml_path: Path, img_w: int, img_h: int) -> list[str]:
    """Parse a VOC XML file and return YOLO-format label lines."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return []

    # Try to read image size from XML if not provided
    if img_w <= 0 or img_h <= 0:
        size = root.find("size")
        if size is not None:
            img_w = int(size.find("width").text)
            img_h = int(size.find("height").text)

    if img_w <= 0 or img_h <= 0:
        return []

    lines = []
    for obj in root.iter("object"):
        name = obj.find("name")
        if name is None:
            continue
        cls = name.text.strip().lower()
        if cls not in CLASSES:
            continue
        cls_id = CLASSES.index(cls)

        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        try:
            x1 = float(bndbox.find("xmin").text)
            y1 = float(bndbox.find("ymin").text)
            x2 = float(bndbox.find("xmax").text)
            y2 = float(bndbox.find("ymax").text)
        except (TypeError, AttributeError):
            continue

        cx = (x1 + x2) / 2.0 / img_w
        cy = (y1 + y2) / 2.0 / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h

        # Clamp to [0, 1]
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        bw = max(0.0, min(1.0, bw))
        bh = max(0.0, min(1.0, bh))

        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Collect file lists
# ─────────────────────────────────────────────────────────────────────────────
def collect_split(split_name: str, img_subdir: str):
    """
    Returns a list of (img_path, xml_path) pairs for one split.

    Args:
        split_name : "train" or "test"
        img_subdir : subdirectory name inside the split (e.g. "VOC2007-FOG")
    """
    split_dir  = VOC_FOG / split_name
    img_dir    = split_dir / img_subdir
    annot_dir  = split_dir / "Annotations"

    pairs = []
    for img_file in sorted(img_dir.iterdir()):
        if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        xml_file = annot_dir / (img_file.stem + ".xml")
        if xml_file.exists():
            pairs.append((img_file, xml_file))

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Write one split: copy images + convert labels
# ─────────────────────────────────────────────────────────────────────────────
def write_split(pairs: list, split_out: str, total: int, start: int = 0):
    """
    Copy images and write YOLO labels for a list of (img_path, xml_path) pairs.

    Args:
        pairs     : list of (img_path, xml_path)
        split_out : "train", "val", or "test"
        total     : total pairs being processed (for progress display)
        start     : starting count offset for progress display
    """
    img_out_dir = OUTPUT / "images" / split_out
    lbl_out_dir = OUTPUT / "labels" / split_out
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    for i, (img_path, xml_path) in enumerate(pairs):
        # Convert annotations
        lines = voc_to_yolo(xml_path, -1, -1)  # let XML provide size
        if not lines:
            skipped += 1
            continue

        # Copy foggy image
        dst_img = img_out_dir / img_path.name
        shutil.copy2(img_path, dst_img)

        # Write YOLO label
        dst_lbl = lbl_out_dir / (img_path.stem + ".txt")
        dst_lbl.write_text("\n".join(lines) + "\n")
        written += 1

        if (start + i + 1) % 500 == 0:
            print(f"  [{start + i + 1}/{total}] processed...")

    return written, skipped


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    random.seed(SEED)

    print("=" * 55)
    print(" VOC-FOG → YOLO Converter for SAID Stage 1 Pre-training")
    print("=" * 55)
    print(f"\n  Source : {VOC_FOG}")
    print(f"  Output : {OUTPUT}\n")

    # --- Train split (9,578 images) ---
    print("[1/3] Collecting train pairs from VOC2007-FOG ...")
    train_pairs = collect_split("train", "VOC2007-FOG")
    print(f"      Found {len(train_pairs)} image-annotation pairs.")

    # --- Test split → split into val + test ---
    print("[2/3] Collecting test pairs from VOCtest-FOG ...")
    test_pairs = collect_split("test", "VOCtest-FOG")
    random.shuffle(test_pairs)
    n_val = int(len(test_pairs) * VAL_RATIO)
    val_pairs  = test_pairs[:n_val]
    test_pairs = test_pairs[n_val:]
    print(f"      Found {len(test_pairs) + len(val_pairs)} total test pairs.")
    print(f"      → val: {len(val_pairs)}, test: {len(test_pairs)}")

    total = len(train_pairs) + len(val_pairs) + len(test_pairs)

    # --- Write train ---
    print(f"\n[3/3] Converting and copying {total} samples...")
    print("  Writing train ...")
    n_train, sk_train = write_split(train_pairs, "train", total, 0)

    print("  Writing val ...")
    n_val_w, sk_val = write_split(val_pairs, "val", total, len(train_pairs))

    print("  Writing test ...")
    n_test, sk_test = write_split(
        test_pairs, "test", total, len(train_pairs) + len(val_pairs)
    )

    # --- Write vocfog.yaml ---
    yaml_content = (
        f"path: {OUTPUT.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n\n"
        f"names:\n"
        f"  0: person\n"
        f"  1: bicycle\n"
        f"  2: car\n"
        f"  3: bus\n"
        f"  4: motorbike\n"
    )
    YAML_PATH.write_text(yaml_content)

    print("\n" + "=" * 55)
    print(" Summary")
    print("=" * 55)
    print(f"  Train  : {n_train:,} written  ({sk_train} skipped — no RTTS classes)")
    print(f"  Val    : {n_val_w:,} written  ({sk_val} skipped)")
    print(f"  Test   : {n_test:,} written  ({sk_test} skipped)")
    print(f"\n  vocfog.yaml → {YAML_PATH}")
    print(f"  Dataset   → {OUTPUT}")
    print("\nRun Stage 1 training with:")
    print("  python train.py --stage vocfog  (or --stage both)")


if __name__ == "__main__":
    main()
