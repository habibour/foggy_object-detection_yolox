"""
smoke_test.py — Quick end-to-end pipeline check for SAID
=========================================================
Tests (in order, each can pass independently):
  1. Python imports      — all source files parse cleanly
  2. Dataset paths       — RTTS_Ready & VOC_FOG_YOLO exist with correct structure
  3. YAML configs        — rtts.yaml & vocfog.yaml point to real directories
  4. Custom modules      — A2C2f_FSA & WIoU_v3_InnerMPDIoU forward passes (needs torch)
  5. Mini train run      — 2 epochs on 20 RTTS images to validate the full pipeline (needs ultralytics)

Run with:
    python smoke_test.py             # all tests
    python smoke_test.py --skip-train  # skip the mini training run
"""

import argparse
import sys
import os
import ast
from pathlib import Path

ROOT = Path(__file__).parent
PASS = "✅"
FAIL = "❌"
SKIP = "⚠️ "


def section(title):
    print(f"\n{'─'*55}")
    print(f" {title}")
    print(f"{'─'*55}")


def ok(msg):   print(f"  {PASS}  {msg}")
def fail(msg): print(f"  {FAIL}  {msg}"); return False
def skip(msg): print(f"  {SKIP}  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Syntax check all source files
# ─────────────────────────────────────────────────────────────────────────────
def test_syntax():
    section("TEST 1 — Python Syntax Check")
    files = [
        ROOT / "prepare_data.py",
        ROOT / "prepare_vocfog.py",
        ROOT / "train.py",
        ROOT / "said" / "__init__.py",
        ROOT / "said" / "a2c2f_fsa.py",
        ROOT / "said" / "wiou_loss.py",
    ]
    passed = True
    for f in files:
        if not f.exists():
            fail(f"{f.name} — FILE NOT FOUND")
            passed = False
            continue
        try:
            ast.parse(f.read_text())
            ok(f"{f.relative_to(ROOT)}")
        except SyntaxError as e:
            fail(f"{f.name} — SyntaxError: {e}")
            passed = False
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Dataset directory structure
# ─────────────────────────────────────────────────────────────────────────────
def test_dataset_structure():
    section("TEST 2 — Dataset Directory Structure")
    passed = True

    checks = {
        "RTTS_Ready/images/train": ROOT / "RTTS_Ready" / "images" / "train",
        "RTTS_Ready/images/val":   ROOT / "RTTS_Ready" / "images" / "val",
        "RTTS_Ready/images/test":  ROOT / "RTTS_Ready" / "images" / "test",
        "RTTS_Ready/labels/train": ROOT / "RTTS_Ready" / "labels" / "train",
        "RTTS_Ready/labels/val":   ROOT / "RTTS_Ready" / "labels" / "val",
        "RTTS_Ready/labels/test":  ROOT / "RTTS_Ready" / "labels" / "test",
        "VOC_FOG_YOLO/images/train": ROOT / "VOC_FOG_YOLO" / "images" / "train",
        "VOC_FOG_YOLO/images/val":   ROOT / "VOC_FOG_YOLO" / "images" / "val",
        "VOC_FOG_YOLO/images/test":  ROOT / "VOC_FOG_YOLO" / "images" / "test",
        "VOC_FOG_YOLO/labels/train": ROOT / "VOC_FOG_YOLO" / "labels" / "train",
    }

    for label, path in checks.items():
        if path.exists():
            n = len(list(path.iterdir()))
            ok(f"{label}  ({n} files)")
        else:
            fail(f"{label}  — MISSING")
            passed = False

    return passed


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: YAML config validation
# ─────────────────────────────────────────────────────────────────────────────
def test_yaml_configs():
    section("TEST 3 — YAML Config Files")
    try:
        import yaml
    except ImportError:
        skip("PyYAML not installed — parsing manually")
        yaml = None

    passed = True
    for yaml_path in [ROOT / "rtts.yaml", ROOT / "vocfog.yaml"]:
        if not yaml_path.exists():
            fail(f"{yaml_path.name} — NOT FOUND")
            passed = False
            continue

        content = yaml_path.read_text()

        # Basic field checks without yaml lib
        for field in ["path:", "train:", "val:", "names:"]:
            if field not in content:
                fail(f"{yaml_path.name} — missing field: {field}")
                passed = False

        # Check that the dataset root path actually exists
        for line in content.splitlines():
            if line.strip().startswith("path:"):
                dataset_path = Path(line.split(":", 1)[1].strip())
                if dataset_path.exists():
                    ok(f"{yaml_path.name} → path exists: {dataset_path}")
                else:
                    fail(f"{yaml_path.name} → path NOT found: {dataset_path}")
                    passed = False
                break

    return passed


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Custom module forward passes
# ─────────────────────────────────────────────────────────────────────────────
def test_custom_modules():
    section("TEST 4 — Custom Module Forward Passes")

    try:
        import torch
    except ImportError:
        skip("PyTorch not installed — skipping module tests")
        return None   # None = skipped

    sys.path.insert(0, str(ROOT))

    try:
        from said.a2c2f_fsa import A2C2f_FSA, FrequencyDisentangleModule, DeformableSpatialAttention
        from said.wiou_loss import WIoU_v3_InnerMPDIoU, WIoUv3Loss, inner_mpdIoU_loss
    except Exception as e:
        fail(f"Import error: {e}")
        return False

    passed = True

    # FrequencyDisentangleModule
    try:
        fdm = FrequencyDisentangleModule(channels=32)
        x = torch.randn(2, 32, 64, 64)
        out = fdm(x)
        assert out.shape == x.shape
        ok(f"FrequencyDisentangleModule: {tuple(x.shape)} → {tuple(out.shape)}")
    except Exception as e:
        fail(f"FrequencyDisentangleModule: {e}")
        passed = False

    # DeformableSpatialAttention
    try:
        dsa = DeformableSpatialAttention(channels=32, num_heads=4)
        x = torch.randn(2, 32, 32, 32)
        out = dsa(x)
        assert out.shape == x.shape
        ok(f"DeformableSpatialAttention: {tuple(x.shape)} → {tuple(out.shape)}")
    except Exception as e:
        fail(f"DeformableSpatialAttention: {e}")
        passed = False

    # A2C2f_FSA (full module)
    try:
        m = A2C2f_FSA(in_channels=64, out_channels=64, n_blocks=2, use_dsa=True)
        x = torch.randn(2, 64, 80, 80)
        out = m(x)
        assert out.shape == (2, 64, 80, 80)
        ok(f"A2C2f_FSA (n_blocks=2, DSA=True): {tuple(x.shape)} → {tuple(out.shape)}")
    except Exception as e:
        fail(f"A2C2f_FSA: {e}")
        passed = False

    # WIoUv3Loss
    try:
        loss_fn = WIoUv3Loss(momentum=0.5)
        pred   = torch.tensor([[10., 10., 50., 50.], [20., 20., 80., 80.]])
        target = torch.tensor([[12., 12., 52., 52.], [25., 25., 85., 85.]])
        loss_fn.train()
        loss = loss_fn(pred, target)
        assert loss.item() >= 0
        ok(f"WIoUv3Loss: loss = {loss.item():.5f}")
    except Exception as e:
        fail(f"WIoUv3Loss: {e}")
        passed = False

    # inner_mpdIoU_loss
    try:
        loss = inner_mpdIoU_loss(pred, target, inner_scale=0.7)
        assert loss.shape == (2,)
        ok(f"inner_mpdIoU_loss: mean = {loss.mean().item():.5f}")
    except Exception as e:
        fail(f"inner_mpdIoU_loss: {e}")
        passed = False

    # Combined loss
    try:
        combined = WIoU_v3_InnerMPDIoU(alpha=0.6, inner_scale=0.7)
        combined.train()
        loss = combined(pred, target)
        assert loss.item() >= 0
        ok(f"WIoU_v3_InnerMPDIoU (combined): loss = {loss.item():.5f}")
    except Exception as e:
        fail(f"WIoU_v3_InnerMPDIoU: {e}")
        passed = False

    # Gradient flow check
    try:
        pred_g = pred.clone().requires_grad_(True)
        loss = combined(pred_g, target)
        loss.backward()
        assert pred_g.grad is not None
        ok(f"Gradient flow: ✓ gradients flow through combined loss")
    except Exception as e:
        fail(f"Gradient flow: {e}")
        passed = False

    return passed


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Mini training run (2 epochs, tiny dataset subset)
# ─────────────────────────────────────────────────────────────────────────────
def test_mini_train(args):
    section("TEST 5 — Mini Training Run (2 epochs)")

    try:
        from ultralytics import YOLO
        import torch
    except ImportError:
        skip("ultralytics not installed — skipping mini train test")
        skip("Install with: pip install ultralytics torch torchvision")
        return None

    # Auto-detect best device
    if args.device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "0"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"  Device: {device}")

    rtts_train_imgs = ROOT / "RTTS_Ready" / "images" / "train"
    if not rtts_train_imgs.exists():
        fail("RTTS_Ready not found — run: python prepare_data.py first")
        return False

    # Create a tiny dataset YAML pointing at first 20 images only
    mini_dir     = ROOT / "_smoke_test_mini"
    mini_img_dir = mini_dir / "images" / "train"
    mini_lbl_dir = mini_dir / "labels" / "train"
    mini_img_val = mini_dir / "images" / "val"
    mini_lbl_val = mini_dir / "labels" / "val"

    for d in [mini_img_dir, mini_lbl_dir, mini_img_val, mini_lbl_val]:
        d.mkdir(parents=True, exist_ok=True)

    import shutil

    # Copy 20 train images + labels
    imgs = sorted(rtts_train_imgs.iterdir())[:20]
    for img in imgs:
        shutil.copy2(img, mini_img_dir / img.name)
        lbl = ROOT / "RTTS_Ready" / "labels" / "train" / (img.stem + ".txt")
        if lbl.exists():
            shutil.copy2(lbl, mini_lbl_dir / lbl.name)

    # Copy 5 val images + labels for val split
    val_imgs = sorted((ROOT / "RTTS_Ready" / "images" / "val").iterdir())[:5]
    for img in val_imgs:
        shutil.copy2(img, mini_img_val / img.name)
        lbl = ROOT / "RTTS_Ready" / "labels" / "val" / (img.stem + ".txt")
        if lbl.exists():
            shutil.copy2(lbl, mini_lbl_val / lbl.name)

    # Write mini YAML
    mini_yaml = mini_dir / "mini_rtts.yaml"
    mini_yaml.write_text(
        f"path: {mini_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"names:\n"
        f"  0: person\n"
        f"  1: bicycle\n"
        f"  2: car\n"
        f"  3: bus\n"
        f"  4: motorbike\n"
    )

    print(f"  Mini dataset: 20 train + 5 val images")
    print(f"  Running 2 epochs on device={args.device} ...")

    try:
        model = YOLO("yolov12n.pt")   # use nano for speed in smoke test
        results = model.train(
            data    = str(mini_yaml),
            epochs  = 2,
            batch   = 4,
            imgsz   = 320,
            device  = device,
            project = str(ROOT / "runs" / "smoke_test"),
            name    = "mini_run",
            val     = True,
            plots   = False,
            verbose = False,
            save    = False,
        )
        map50 = results.results_dict.get("metrics/mAP50(B)", 0)
        ok(f"Mini train completed — mAP50={map50:.4f}")
        ok(f"Ultralytics training pipeline: fully functional")
    except Exception as e:
        fail(f"Mini train failed: {e}")
        return False
    finally:
        # Clean up mini dataset
        shutil.rmtree(mini_dir, ignore_errors=True)
        shutil.rmtree(ROOT / "runs" / "smoke_test", ignore_errors=True)

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-train", action="store_true",
                   help="Skip the mini training run (Test 5)")
    p.add_argument("--device", type=str, default=None,
                   help="Device: 'mps', '0' (CUDA), 'cpu'. Auto-detected if not set.")
    args = p.parse_args()

    print("\n" + "═"*55)
    print(" SAID Smoke Test")
    print("═"*55)

    results = {}
    results["syntax"]   = test_syntax()
    results["dataset"]  = test_dataset_structure()
    results["yaml"]     = test_yaml_configs()
    results["modules"]  = test_custom_modules()

    if args.skip_train:
        skip("Mini train skipped (--skip-train)")
        results["mini_train"] = None
    else:
        results["mini_train"] = test_mini_train(args)

    # ── Summary ──────────────────────────────────────────────────────────────
    section("SUMMARY")
    all_passed = True
    for name, result in results.items():
        if result is True:
            ok(f"{name}")
        elif result is None:
            skip(f"{name}  (skipped)")
        else:
            fail(f"{name}  — FAILED")
            all_passed = False

    print()
    if all_passed:
        print("  🎉 All tests passed! Ready to run: python train.py --stage both")
    else:
        print("  ⚠️  Some tests failed. Fix the issues above before training.")
    print()


if __name__ == "__main__":
    main()
