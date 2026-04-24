"""
SAID Training Script — Two-Stage Fine-Tuning via Ultralytics API
=================================================================
Stage 1: Pre-train on VOC-FOG  (synthetic foggy, 9,578 train images)
Stage 2: Fine-tune on RTTS     (real-world foggy, 4,322 images, 60/20/20)

Architecture highlights:
  - Base: YOLO11-X pretrained weights (yolo11x.pt)
  - Preprocessing  → A2C2f-FSA (FFT frequency disentanglement + DSA)
  - Loss           → WIoU_v3_InnerMPDIoU (difficulty-adaptive weighting)

Custom Evaluation & Checkpoint Schedule (Stage 2):
  - Validation   : every 10 epochs  (RTTS val split)
  - Test         : every 20 epochs  (RTTS test + VOC-FOG test)
  - Best save    : every  5 epochs  (replaces rolling best if mAP improved)
  - Final save   : on training end  (always saved)

Usage (local):
    python train.py --stage both
    python train.py --stage rtts
    python train.py --stage validate

Usage (Kaggle P100 — auto-detected):
    python train.py --stage both --kaggle
"""

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────────────────────
# Environment Detection
# ─────────────────────────────────────────────────────────────────────────────
IS_KAGGLE = os.path.exists("/kaggle/working")


def auto_device() -> str:
    """Return the best available device string."""
    try:
        import torch
        if torch.cuda.is_available():
            return "0"               # CUDA (Kaggle P100 / any NVIDIA)
        if torch.backends.mps.is_available():
            return "mps"             # Apple Silicon
    except Exception:
        pass
    return "cpu"


def get_paths(kaggle: bool = False):
    """Return (ROOT, RTTS_YAML, VOCFOG_YAML, WEIGHTS_DIR, CKPT_DIR)."""
    if kaggle or IS_KAGGLE:
        ROOT        = Path("/kaggle/working")
        RTTS_YAML   = ROOT / "rtts.yaml"
        VOCFOG_YAML = ROOT / "vocfog.yaml"
        WEIGHTS_DIR = ROOT / "weights"
        CKPT_DIR    = ROOT / "checkpoints"
    else:
        ROOT        = Path(__file__).parent
        RTTS_YAML   = ROOT / "rtts.yaml"
        VOCFOG_YAML = ROOT / "vocfog.yaml"
        WEIGHTS_DIR = ROOT / "weights"
        CKPT_DIR    = ROOT / "checkpoints"

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    return ROOT, RTTS_YAML, VOCFOG_YAML, WEIGHTS_DIR, CKPT_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters (shared base — overridden per stage below)
# ─────────────────────────────────────────────────────────────────────────────
def build_common_args(batch: int, workers: int = 4) -> dict:
    return dict(
        imgsz         = 640,
        optimizer     = "AdamW",
        lr0           = 0.01,
        lrf           = 0.01,
        weight_decay  = 0.0005,
        warmup_epochs = 3.0,
        warmup_bias_lr= 0.1,
        hsv_h         = 0.015,
        hsv_s         = 0.7,
        hsv_v         = 0.4,
        flipud        = 0.0,
        fliplr        = 0.5,
        mosaic        = 1.0,
        mixup         = 0.15,
        copy_paste    = 0.1,
        degrees       = 0.0,
        translate     = 0.1,
        scale         = 0.5,
        shear         = 0.0,
        perspective   = 0.0,
        close_mosaic  = 10,
        amp           = True,       # FP16 mixed precision (P100 ~2× speedup)
        workers       = workers,
        plots         = True,
        verbose       = True,
        val           = True,
        save          = True,
        save_period   = -1,         # we handle checkpointing via callbacks
    )


# ─────────────────────────────────────────────────────────────────────────────
# Callback Factory
# ─────────────────────────────────────────────────────────────────────────────
def make_callbacks(args, run_dir: Path, rtts_yaml: str, vocfog_yaml: str):
    """
    Custom eval + checkpoint callbacks for Stage 2 Phase 2b.

    Schedule:
      - Val  every args.val_freq  epochs  → RTTS val
      - Test every args.test_freq epochs  → RTTS test + VOC-FOG test
      - Save rolling best every args.save_freq epochs (if mAP improved)
      - Save final weights on training end
    """
    WEIGHTS_DIR = run_dir.parent.parent / "weights"
    CKPT_DIR    = run_dir.parent.parent / "checkpoints"
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    ROLLING_BEST   = CKPT_DIR   / "said_rolling_best.pt"
    STAGE2_WEIGHTS = WEIGHTS_DIR / "said_rtts_final.pt"
    LOG_PATH       = run_dir / "eval_log.csv"

    state = {"best_map50": 0.0, "best_epoch": -1}

    def _run_val(tag: str, epoch: int, split: str, yaml: str) -> dict:
        last_pt = run_dir / "weights" / "last.pt"
        m = YOLO(str(last_pt)) if last_pt.exists() else None
        if m is None:
            return {}
        metrics = m.val(
            data=yaml, imgsz=640, batch=args.batch,
            device=args.device, split=split,
            plots=False, verbose=False,
        )
        row = {
            "epoch":     epoch,
            "tag":       tag,
            "split":     split,
            "map50":     round(metrics.box.map50, 4),
            "map50_95":  round(metrics.box.map,   4),
            "precision": round(metrics.box.mp,    4),
            "recall":    round(metrics.box.mr,    4),
        }
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        write_hdr = not LOG_PATH.exists()
        with open(LOG_PATH, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_hdr:
                w.writeheader()
            w.writerow(row)
        print(
            f"  [{tag}|ep{epoch:03d}]  "
            f"mAP50={row['map50']:.4f}  mAP50:95={row['map50_95']:.4f}  "
            f"P={row['precision']:.4f}  R={row['recall']:.4f}"
        )
        return row

    def _maybe_save_best(epoch: int, map50: float):
        """Save rolling best checkpoint every save_freq epochs if mAP improved."""
        if epoch % args.save_freq != 0:
            return
        if map50 > state["best_map50"]:
            state["best_map50"] = map50
            state["best_epoch"] = epoch
            last_pt = run_dir / "weights" / "last.pt"
            if last_pt.exists():
                shutil.copy2(last_pt, ROLLING_BEST)
                print(
                    f"  ★ Rolling best updated: mAP50={map50:.4f} "
                    f"@ epoch {epoch} → {ROLLING_BEST.name}"
                )

    def on_fit_epoch_end(trainer):
        epoch = trainer.epoch + 1

        # ── Validation every val_freq epochs ──────────────────────────────
        if epoch % args.val_freq == 0:
            print(f"\n{'─'*55}\n VALIDATION @ epoch {epoch}\n{'─'*55}")
            row = _run_val("VAL-RTTS", epoch, "val", rtts_yaml)
            if row:
                _maybe_save_best(epoch, row["map50"])

        # ── Test every test_freq epochs (RTTS + VOC-FOG) ──────────────────
        if epoch % args.test_freq == 0:
            print(f"\n{'─'*55}\n TEST @ epoch {epoch}\n{'─'*55}")
            _run_val("TEST-RTTS",   epoch, "test", rtts_yaml)
            if Path(vocfog_yaml).exists():
                _run_val("TEST-VOCFOG", epoch, "test", vocfog_yaml)

    def on_train_end(trainer):
        epoch = trainer.epoch + 1
        print(f"\n{'═'*55}\n TRAINING COMPLETE\n{'═'*55}")

        # Final evaluation
        print("\n Final RTTS val:")
        _run_val("FINAL-VAL-RTTS",    epoch, "val",  rtts_yaml)
        print("\n Final RTTS test:")
        _run_val("FINAL-TEST-RTTS",   epoch, "test", rtts_yaml)
        if Path(vocfog_yaml).exists():
            print("\n Final VOC-FOG test:")
            _run_val("FINAL-TEST-VOCFOG", epoch, "test", vocfog_yaml)

        # Save final weights
        best_pt = run_dir / "weights" / "best.pt"
        last_pt = run_dir / "weights" / "last.pt"
        if best_pt.exists():
            shutil.copy2(best_pt, STAGE2_WEIGHTS)
            print(f"\n  ✓ Final (best.pt)  → {STAGE2_WEIGHTS}")
        if last_pt.exists():
            last_final = WEIGHTS_DIR / "said_rtts_last.pt"
            shutil.copy2(last_pt, last_final)
            print(f"  ✓ Final (last.pt)  → {last_final}")

        # Summary JSON
        summary = {
            "best_rolling_map50": state["best_map50"],
            "best_rolling_epoch": state["best_epoch"],
            "rolling_best":       str(ROLLING_BEST),
            "final_weights":      str(STAGE2_WEIGHTS),
            "eval_log":           str(LOG_PATH),
        }
        (run_dir / "said_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\n  Rolling best: mAP50={state['best_map50']:.4f} @ ep{state['best_epoch']}")
        print(f"  Eval log    : {LOG_PATH}")

    return {"on_fit_epoch_end": on_fit_epoch_end, "on_train_end": on_train_end}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Pre-train on VOC-FOG
# ─────────────────────────────────────────────────────────────────────────────
def stage1_vocfog(args, paths):
    ROOT, RTTS_YAML, VOCFOG_YAML, WEIGHTS_DIR, CKPT_DIR = paths
    STAGE1_WEIGHTS = WEIGHTS_DIR / "said_vocfog_pretrained.pt"

    if not VOCFOG_YAML.exists():
        raise FileNotFoundError(f"VOC-FOG config not found: {VOCFOG_YAML}")

    print("\n" + "=" * 55)
    print(" SAID — Stage 1: Pre-training on VOC-FOG")
    print("=" * 55)

    common = build_common_args(args.batch)
    model  = YOLO("yolo11x.pt")
    results = model.train(
        data    = str(VOCFOG_YAML),
        epochs  = args.s1_epochs,
        batch   = args.batch,
        device  = args.device,
        project = str(ROOT / "runs" / "said"),
        name    = "stage1_vocfog",
        patience= 15,
        **common,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    shutil.copy2(best, STAGE1_WEIGHTS)
    print(f"\nStage 1 complete → {STAGE1_WEIGHTS}")
    return str(STAGE1_WEIGHTS)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Fine-tune on RTTS
# ─────────────────────────────────────────────────────────────────────────────
def stage2_rtts(args, paths, init_weights: str = None):
    ROOT, RTTS_YAML, VOCFOG_YAML, WEIGHTS_DIR, CKPT_DIR = paths
    STAGE1_WEIGHTS = WEIGHTS_DIR / "said_vocfog_pretrained.pt"

    if init_weights is None:
        init_weights = str(STAGE1_WEIGHTS) if STAGE1_WEIGHTS.exists() else "yolo11x.pt"
    print(f"\nInitialising Stage 2 from: {init_weights}")

    print("\n" + "=" * 55)
    print(" SAID — Stage 2: Fine-tuning on RTTS")
    print("=" * 55)

    # ── Phase 2a: Freeze backbone (10 epochs) ─────────────────────────────
    print("\n[Phase 2a] Backbone frozen — head warmup on RTTS (10 epochs)")
    model_a = YOLO(init_weights)
    model_a.train(
        data         = str(RTTS_YAML),
        epochs       = 10,
        batch        = args.batch,
        device       = args.device,
        project      = str(ROOT / "runs" / "said"),
        name         = "stage2a_freeze",
        freeze       = list(range(10)),
        lr0          = 0.001,
        lrf          = 0.1,
        weight_decay = 0.0005,
        warmup_epochs= 1.0,
        imgsz        = 640,
        optimizer    = "AdamW",
        mosaic       = 0.8,
        amp          = True,
        workers      = 4,
        val          = True,
        save         = True,
        plots        = True,
        verbose      = True,
    )

    phase2a_best = ROOT / "runs" / "said" / "stage2a_freeze" / "weights" / "best.pt"
    if not phase2a_best.exists():
        print(f"  Warning: phase2a best not found, using {init_weights}")
        phase2a_best = Path(init_weights)

    # ── Phase 2b: Full fine-tune with custom callbacks ─────────────────────
    print("\n[Phase 2b] Full fine-tune on RTTS with custom eval schedule")
    run_name = "stage2b_full"
    run_dir  = ROOT / "runs" / "said" / run_name

    common  = build_common_args(args.batch)
    model_b = YOLO(str(phase2a_best))

    callbacks = make_callbacks(
        args       = args,
        run_dir    = run_dir,
        rtts_yaml  = str(RTTS_YAML),
        vocfog_yaml= str(VOCFOG_YAML),
    )
    for event, fn in callbacks.items():
        model_b.add_callback(event, fn)

    model_b.train(
        data    = str(RTTS_YAML),
        epochs  = args.epochs,
        batch   = args.batch,
        device  = args.device,
        project = str(ROOT / "runs" / "said"),
        name    = run_name,
        patience= 0,                   # we manage checkpoints via callbacks
        **{**common, "val": False},    # disable default per-epoch val
    )
    print(f"\nStage 2 complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────
def validate(args, paths):
    ROOT, RTTS_YAML, VOCFOG_YAML, WEIGHTS_DIR, _ = paths
    weights = args.weights or str(WEIGHTS_DIR / "said_rtts_final.pt")
    if not Path(weights).exists():
        print(f"Weights not found: {weights}")
        return

    print(f"\n{'═'*55}\n SAID — Final Evaluation\n{'═'*55}")
    model = YOLO(weights)

    def _eval(label, yaml, split):
        m = model.val(data=yaml, imgsz=640, batch=args.batch,
                      device=args.device, split=split, plots=True, verbose=False)
        print(f"\n── {label} ({split}) ──")
        print(f"  mAP@0.5      : {m.box.map50:.4f}")
        print(f"  mAP@0.5:0.95 : {m.box.map:.4f}")
        print(f"  Precision    : {m.box.mp:.4f}")
        print(f"  Recall       : {m.box.mr:.4f}")

    _eval("RTTS val",    str(RTTS_YAML),   "val")
    _eval("RTTS test",   str(RTTS_YAML),   "test")
    if VOCFOG_YAML.exists():
        _eval("VOC-FOG test", str(VOCFOG_YAML), "test")


# ─────────────────────────────────────────────────────────────────────────────
# Sanity Check
# ─────────────────────────────────────────────────────────────────────────────
def sanity_check():
    import torch
    from said.a2c2f_fsa import A2C2f_FSA
    from said.wiou_loss import WIoU_v3_InnerMPDIoU

    print("\nRunning sanity checks...")
    m   = A2C2f_FSA(in_channels=64, out_channels=64, n_blocks=2, use_dsa=True)
    x   = torch.randn(2, 64, 80, 80)
    out = m(x)
    assert out.shape == (2, 64, 80, 80)
    print(f"  ✓ A2C2f_FSA  : {tuple(x.shape)} → {tuple(out.shape)}")

    loss_fn = WIoU_v3_InnerMPDIoU(alpha=0.6, inner_scale=0.7)
    pred    = torch.tensor([[10., 10., 50., 50.], [20., 20., 80., 80.]])
    target  = torch.tensor([[12., 12., 52., 52.], [15., 15., 75., 75.]])
    loss_fn.train()
    loss = loss_fn(pred, target)
    assert loss.item() >= 0
    print(f"  ✓ WIoU_v3_InnerMPDIoU : loss = {loss.item():.4f}")
    print("All sanity checks passed!\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="SAID Training Script")
    p.add_argument("--stage",     choices=["vocfog","rtts","both","validate","check"],
                   default="both")
    p.add_argument("--epochs",    type=int, default=100,
                   help="Stage 2 Phase 2b epochs")
    p.add_argument("--s1-epochs", type=int, default=50,
                   help="Stage 1 VOC-FOG pre-training epochs")
    p.add_argument("--batch",     type=int, default=None,
                   help="Batch size (auto: 32 on Kaggle/CUDA, 16 local)")
    p.add_argument("--device",    type=str, default=None,
                   help="Device: '0' (CUDA/P100), 'mps', 'cpu'. Auto-detected.")
    p.add_argument("--weights",   type=str, default=None,
                   help="Weights path for --stage validate")
    p.add_argument("--kaggle",    action="store_true",
                   help="Force Kaggle mode (auto-detected if /kaggle/working exists)")
    p.add_argument("--val-freq",  type=int, default=10,
                   help="Validate every N epochs (default: 10)")
    p.add_argument("--test-freq", type=int, default=20,
                   help="Test every N epochs on RTTS+VOC-FOG (default: 20)")
    p.add_argument("--save-freq", type=int, default=5,
                   help="Save rolling best every N epochs (default: 5)")
    return p.parse_args()


def main():
    args    = parse_args()
    kaggle  = args.kaggle or IS_KAGGLE
    paths   = get_paths(kaggle)

    # Auto-fill device and batch based on environment
    if args.device is None:
        args.device = auto_device()
    if args.batch is None:
        args.batch = 32 if (kaggle or args.device == "0") else 16

    print(f"\n{'═'*55}")
    print(f" SAID Training Configuration")
    print(f"{'═'*55}")
    print(f"  Environment : {'Kaggle P100' if kaggle else 'Local'}")
    print(f"  Device      : {args.device}")
    print(f"  Batch size  : {args.batch}")
    print(f"  Stage       : {args.stage}")
    print(f"  Epochs (S2) : {args.epochs}")
    print(f"  Val every   : {args.val_freq} epochs")
    print(f"  Test every  : {args.test_freq} epochs")
    print(f"  Save best   : every {args.save_freq} epochs")
    print(f"{'═'*55}\n")

    if args.stage == "check":
        sanity_check()
        return
    if args.stage == "validate":
        validate(args, paths)
        return

    stage1_weights = None
    if args.stage in ("vocfog", "both"):
        stage1_weights = stage1_vocfog(args, paths)

    if args.stage in ("rtts", "both"):
        stage2_rtts(args, paths, init_weights=stage1_weights)

    if args.stage in ("rtts", "both"):
        validate(args, paths)


if __name__ == "__main__":
    main()
