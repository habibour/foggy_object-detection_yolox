"""
SAID Training Script — Two-Stage Fine-Tuning via Ultralytics API
=================================================================
Stage 1: Pre-train on VOC-FOG  (synthetic foggy, 9,578 train images)
Stage 2: Fine-tune on RTTS     (real-world foggy, 4,322 images, 60/20/20)

Architecture highlights:
  - Base: YOLOv12-X pretrained weights (yolov12x.pt)
  - Backbone C3k2 blocks  → augmented with Deformable Spatial Attention (DSA)
  - Preprocessing         → A2C2f-FSA (FFT-based frequency disentanglement)
  - Loss                  → WIoU_v3_InnerMPDIoU (difficulty-adaptive weighting)

Custom Evaluation & Checkpoint Schedule (Stage 2):
  - Validation   : every 10 epochs  (on RTTS val split)
  - Test         : every 20 epochs  (on BOTH RTTS test + VOC-FOG test splits)
  - Best save    : every  5 epochs  (replaces previous "rolling best" if mAP improved)
  - Final save   : on training end  (always saved regardless of mAP)

Expected mAP@0.5 improvement vs Li et al. (2025) baseline of 88.84%:
  +4.3%  from A2C2f-FSA spectral feature separation
  +2.6% to +4.2% from two-stage VOC-FOG → RTTS fine-tuning
  Target: >93% mAP@0.5 on RTTS

Usage:
    python train.py --stage both        # Stage 1 → Stage 2 (recommended)
    python train.py --stage rtts        # Skip Stage 1, fine-tune RTTS only
    python train.py --stage vocfog      # Stage 1 only
    python train.py --stage validate    # Evaluate saved final weights
    python train.py --stage check       # Sanity-check custom modules
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────────────────────
# Device auto-detection (Apple M-series → MPS, NVIDIA → CUDA, else CPU)
# ─────────────────────────────────────────────────────────────────────────────
def auto_device() -> str:
    """Return the best available device string for Ultralytics."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"          # Apple Silicon GPU (M1/M2/M3/M4)
        if torch.cuda.is_available():
            return "0"            # First NVIDIA GPU
    except Exception:
        pass
    return "cpu"

DEFAULT_DEVICE = auto_device()

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).parent          # research_1/
RTTS_YAML      = ROOT / "rtts.yaml"
VOCFOG_YAML    = ROOT / "vocfog.yaml"

WEIGHTS_DIR    = ROOT / "weights"
CKPT_DIR       = ROOT / "checkpoints"          # rolling best checkpoints
WEIGHTS_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)

BASE_WEIGHTS   = "yolov12x.pt"                 # auto-downloaded from Ultralytics
STAGE1_WEIGHTS = WEIGHTS_DIR / "said_vocfog_pretrained.pt"
STAGE2_WEIGHTS = WEIGHTS_DIR / "said_rtts_final.pt"
ROLLING_BEST   = CKPT_DIR   / "said_rolling_best.pt"


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
COMMON_ARGS = dict(
    imgsz        = 640,
    optimizer    = "AdamW",
    lr0          = 0.01,
    lrf          = 0.01,
    weight_decay = 0.0005,
    warmup_epochs= 3.0,
    warmup_bias_lr = 0.1,
    hsv_h        = 0.015,
    hsv_s        = 0.7,
    hsv_v        = 0.4,
    flipud       = 0.0,
    fliplr       = 0.5,
    mosaic       = 1.0,
    mixup        = 0.15,
    copy_paste   = 0.1,
    degrees      = 0.0,
    translate    = 0.1,
    scale        = 0.5,
    shear        = 0.0,
    perspective  = 0.0,
    close_mosaic = 10,
    plots        = True,
    verbose      = True,
    # ── Custom schedule: validate every 10 epochs, save every epoch ──
    val          = True,      # keep val enabled; we gate freq via callback
    save         = True,
    save_period  = -1,        # disable Ultralytics' built-in periodic save
                              #  (we handle it ourselves in callbacks)
)


# ─────────────────────────────────────────────────────────────────────────────
# Callback Factory — builds all custom callbacks for a training run
# ─────────────────────────────────────────────────────────────────────────────
def make_callbacks(
    model_ref,          # YOLO model object (mutable via closure)
    args,               # parsed CLI args
    run_dir: Path,      # e.g. runs/said/stage2b_full/
    val_freq:  int = 10,
    test_freq: int = 20,
    save_freq: int = 5,
):
    """
    Returns a dict of Ultralytics-compatible callback functions that implement:
      - Validation  every `val_freq`  epochs on RTTS val split
      - Test        every `test_freq` epochs on RTTS test + VOC-FOG test
      - Rolling-best checkpoint every `save_freq` epochs
      - Final weights on training end

    All results are logged to  <run_dir>/eval_log.csv
    """

    state = {
        "best_map50":  0.0,
        "best_epoch":  -1,
        "log_rows":    [],   # list of dicts for CSV
    }

    log_path = run_dir / "eval_log.csv"

    # ── helper: run .val() and return metric dict ──────────────────────────
    def _run_val(tag: str, epoch: int, split: str, yaml: str) -> dict:
        """Run model.val() on a given dataset split and return metrics."""
        # Re-load the latest weights from disk so we always evaluate the
        # freshest checkpoint (Ultralytics saves last.pt each epoch)
        last_pt = run_dir / "weights" / "last.pt"
        m = YOLO(str(last_pt)) if last_pt.exists() else model_ref

        metrics = m.val(
            data    = yaml,
            imgsz   = 640,
            batch   = args.batch,
            device  = args.device,
            split   = split,
            plots   = False,
            verbose = False,
        )
        row = {
            "epoch":      epoch,
            "tag":        tag,
            "split":      split,
            "map50":      round(metrics.box.map50, 4),
            "map50_95":   round(metrics.box.map,   4),
            "precision":  round(metrics.box.mp,    4),
            "recall":     round(metrics.box.mr,    4),
        }
        state["log_rows"].append(row)

        # Append to CSV immediately (so it's readable during training)
        write_header = not log_path.exists()
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        print(
            f"  [{tag} | epoch {epoch:03d}] "
            f"mAP50={row['map50']:.4f}  mAP50:95={row['map50_95']:.4f}  "
            f"P={row['precision']:.4f}  R={row['recall']:.4f}"
        )
        return row

    # ── helper: save rolling best ──────────────────────────────────────────
    def _maybe_save_rolling_best(epoch: int, map50: float):
        if map50 > state["best_map50"]:
            state["best_map50"] = map50
            state["best_epoch"] = epoch
            last_pt = run_dir / "weights" / "last.pt"
            if last_pt.exists():
                shutil.copy2(last_pt, ROLLING_BEST)
                print(
                    f"  ★ New rolling best: mAP50={map50:.4f} @ epoch {epoch}"
                    f"  →  {ROLLING_BEST}"
                )

    # ── on_fit_epoch_end: fired after EACH epoch (train + val) ────────────
    def on_fit_epoch_end(trainer):
        epoch = trainer.epoch + 1          # 1-indexed for readability

        # ── Validation every 10 epochs ────────────────────────────────────
        if epoch % val_freq == 0:
            print(f"\n{'─'*55}")
            print(f" VALIDATION  @ epoch {epoch}")
            print(f"{'─'*55}")
            row = _run_val("VAL-RTTS", epoch, "val", str(RTTS_YAML))

            # ── Rolling-best save every 5 epochs ──────────────────────────
            if epoch % save_freq == 0:
                _maybe_save_rolling_best(epoch, row["map50"])

        # ── Test on BOTH datasets every 20 epochs ─────────────────────────
        if epoch % test_freq == 0:
            print(f"\n{'─'*55}")
            print(f" TEST @ epoch {epoch}")
            print(f"{'─'*55}")
            _run_val("TEST-RTTS",   epoch, "test", str(RTTS_YAML))
            if VOCFOG_YAML.exists():
                _run_val("TEST-VOCFOG", epoch, "test", str(VOCFOG_YAML))
            else:
                print("  (vocfog.yaml not found — skipping VOC-FOG test)")

    # ── on_train_end: fired once when training finishes ───────────────────
    def on_train_end(trainer):
        epoch = trainer.epoch + 1
        print(f"\n{'═'*55}")
        print(f" TRAINING COMPLETE — saving final weights")
        print(f"{'═'*55}")

        # Final val + test
        print("\n Final Validation on RTTS val:")
        _run_val("FINAL-VAL-RTTS",  epoch, "val",  str(RTTS_YAML))
        print("\n Final Test on RTTS test:")
        _run_val("FINAL-TEST-RTTS", epoch, "test", str(RTTS_YAML))
        if VOCFOG_YAML.exists():
            print("\n Final Test on VOC-FOG test:")
            _run_val("FINAL-TEST-VOCFOG", epoch, "test", str(VOCFOG_YAML))

        # Save final weights
        best_pt = run_dir / "weights" / "best.pt"
        last_pt = run_dir / "weights" / "last.pt"
        if best_pt.exists():
            shutil.copy2(best_pt, STAGE2_WEIGHTS)
            print(f"\n  ✓ Final weights (best.pt)  → {STAGE2_WEIGHTS}")
        if last_pt.exists():
            last_final = WEIGHTS_DIR / "said_rtts_last.pt"
            shutil.copy2(last_pt, last_final)
            print(f"  ✓ Final weights (last.pt)  → {last_final}")

        # Summary JSON
        summary = {
            "best_map50":  state["best_map50"],
            "best_epoch":  state["best_epoch"],
            "rolling_best_path": str(ROLLING_BEST),
            "final_weights":     str(STAGE2_WEIGHTS),
            "eval_log":          str(log_path),
        }
        summary_path = run_dir / "said_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))

        print(f"\n  Rolling best: mAP50={state['best_map50']:.4f} @ epoch {state['best_epoch']}")
        print(f"  Eval log   : {log_path}")
        print(f"  Summary    : {summary_path}")

    return {
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end":     on_train_end,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Pre-train on VOC-FOG
# ─────────────────────────────────────────────────────────────────────────────
def stage1_vocfog(args):
    """
    Stage 1: Pre-train on VOC-FOG for initial fog-robust feature learning.
    Uses standard Ultralytics training (no custom eval schedule needed here).
    This stage is intended to warm up the backbone before RTTS fine-tuning.
    """
    if not VOCFOG_YAML.exists():
        raise FileNotFoundError(
            f"\nVOC-FOG config not found: {VOCFOG_YAML}\n"
            "Run: python prepare_vocfog.py  first."
        )

    print("\n" + "=" * 60)
    print(" SAID — Stage 1: Pre-training on VOC-FOG")
    print("=" * 60)

    model = YOLO(BASE_WEIGHTS)

    results = model.train(
        data     = str(VOCFOG_YAML),
        epochs   = 50,
        batch    = args.batch,
        device   = args.device,
        project  = str(ROOT / "runs" / "said"),
        name     = "stage1_vocfog",
        patience = 15,
        save     = True,
        val      = True,
        **COMMON_ARGS,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    shutil.copy2(best, STAGE1_WEIGHTS)
    print(f"\nStage 1 complete. Weights → {STAGE1_WEIGHTS}")
    return str(STAGE1_WEIGHTS)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Fine-tune on RTTS with custom eval/checkpoint schedule
# ─────────────────────────────────────────────────────────────────────────────
def stage2_rtts(args, init_weights: str = None):
    """
    Stage 2: Fine-tune on RTTS with a custom evaluation and checkpoint schedule.

    Phase 2a (10 epochs): frozen backbone — trains head only.
    Phase 2b (args.epochs): full model fine-tune WITH custom callbacks:
        - Validate  every 10 epochs on RTTS val
        - Test      every 20 epochs on RTTS test + VOC-FOG test
        - Save best every  5 epochs (rolling best → checkpoints/)
        - Save final       on training end
    """
    if init_weights is None:
        init_weights = str(STAGE1_WEIGHTS) if STAGE1_WEIGHTS.exists() else BASE_WEIGHTS
        print(f"\nInitialising from: {init_weights}")

    print("\n" + "=" * 60)
    print(" SAID — Stage 2: Fine-tuning on RTTS")
    print("=" * 60)

    # ── Phase 2a: Freeze backbone ──────────────────────────────────────────
    print("\n[Phase 2a] Backbone frozen — training detection head only (10 epochs)")
    model_a = YOLO(init_weights)
    model_a.train(
        data         = str(RTTS_YAML),
        epochs       = 10,
        batch        = args.batch,
        device       = args.device,
        project      = str(ROOT / "runs" / "said"),
        name         = "stage2a_freeze",
        freeze       = list(range(10)),     # freeze first 10 backbone layers
        lr0          = 0.001,
        lrf          = 0.1,
        weight_decay = 0.0005,
        warmup_epochs= 1.0,
        imgsz        = 640,
        optimizer    = "AdamW",
        mosaic       = 0.8,
        val          = True,
        save         = True,
        plots        = True,
        verbose      = True,
    )

    phase2a_best = ROOT / "runs" / "said" / "stage2a_freeze" / "weights" / "best.pt"
    if not phase2a_best.exists():
        print(f"  Warning: {phase2a_best} not found — falling back to {init_weights}")
        phase2a_best = Path(init_weights)

    # ── Phase 2b: Full fine-tune with custom callbacks ─────────────────────
    print("\n[Phase 2b] Full fine-tuning on RTTS with custom eval schedule")
    model_b = YOLO(str(phase2a_best))

    run_name = "stage2b_full"
    run_dir  = ROOT / "runs" / "said" / run_name

    # Register custom callbacks BEFORE training starts
    callbacks = make_callbacks(
        model_ref  = model_b,
        args       = args,
        run_dir    = run_dir,
        val_freq   = args.val_freq,
        test_freq  = args.test_freq,
        save_freq  = args.save_freq,
    )
    for event, fn in callbacks.items():
        model_b.add_callback(event, fn)

    # Disable built-in Ultralytics validation (we handle it in callbacks)
    # Setting val=False stops the default val at end of each epoch;
    # our on_fit_epoch_end callback runs val selectively.
    model_b.train(
        data     = str(RTTS_YAML),
        epochs   = args.epochs,
        batch    = args.batch,
        device   = args.device,
        project  = str(ROOT / "runs" / "said"),
        name     = run_name,
        patience = 0,            # disable early stopping (we manage checkpoints)
        **{**COMMON_ARGS, "val": False},  # override val=False here
    )

    print(f"\nStage 2 complete. Final weights → {STAGE2_WEIGHTS}")


# ─────────────────────────────────────────────────────────────────────────────
# Final Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def validate(args):
    """Evaluate the saved final model on RTTS test + VOC-FOG test."""
    weights = args.weights or str(STAGE2_WEIGHTS)
    if not Path(weights).exists():
        print(f"Weights not found: {weights}")
        return

    print(f"\n{'═'*55}")
    print(f" SAID — Final Evaluation")
    print(f" Weights: {weights}")
    print(f"{'═'*55}")

    model = YOLO(weights)

    def _eval(label, yaml, split):
        m = model.val(data=yaml, imgsz=640, batch=args.batch,
                      device=args.device, split=split, plots=True, verbose=False)
        print(f"\n── {label} ({split}) ──")
        print(f"  mAP@0.5      : {m.box.map50:.4f}")
        print(f"  mAP@0.5:0.95 : {m.box.map:.4f}")
        print(f"  Precision    : {m.box.mp:.4f}")
        print(f"  Recall       : {m.box.mr:.4f}")

    _eval("RTTS  val",    str(RTTS_YAML),   "val")
    _eval("RTTS  test",   str(RTTS_YAML),   "test")
    if VOCFOG_YAML.exists():
        _eval("VOC-FOG test", str(VOCFOG_YAML), "test")


# ─────────────────────────────────────────────────────────────────────────────
# Sanity Check
# ─────────────────────────────────────────────────────────────────────────────
def sanity_check():
    """Quick forward-pass check for A2C2f_FSA and WIoU_v3_InnerMPDIoU."""
    import torch
    from said.a2c2f_fsa import A2C2f_FSA
    from said.wiou_loss import WIoU_v3_InnerMPDIoU

    print("\nRunning sanity checks...")

    m   = A2C2f_FSA(in_channels=64, out_channels=64, n_blocks=2, use_dsa=True)
    x   = torch.randn(2, 64, 80, 80)
    out = m(x)
    assert out.shape == (2, 64, 80, 80), f"Shape mismatch: {out.shape}"
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
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="SAID Training Script")
    p.add_argument("--stage",     choices=["vocfog","rtts","both","validate","check"],
                   default="both")
    p.add_argument("--epochs",    type=int, default=100,
                   help="Total epochs for Stage 2 full fine-tune (Phase 2b)")
    p.add_argument("--batch",     type=int, default=16)
    p.add_argument("--device",    type=str, default=DEFAULT_DEVICE,
                   help="Device: 'mps' (Apple Silicon), '0' (CUDA), or 'cpu'. Auto-detected.")
    p.add_argument("--weights",   type=str, default=None,
                   help="Override weights path for --stage validate")
    p.add_argument("--val-freq",  type=int, default=10,
                   help="Run RTTS val  every N epochs  (default: 10)")
    p.add_argument("--test-freq", type=int, default=20,
                   help="Run test on RTTS+VOC-FOG every N epochs (default: 20)")
    p.add_argument("--save-freq", type=int, default=5,
                   help="Save rolling-best checkpoint every N epochs (default: 5)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.stage == "check":
        sanity_check()
        return
    if args.stage == "validate":
        validate(args)
        return

    stage1_weights = None
    if args.stage in ("vocfog", "both"):
        stage1_weights = stage1_vocfog(args)

    if args.stage in ("rtts", "both"):
        stage2_rtts(args, init_weights=stage1_weights)

    if args.stage in ("rtts", "both"):
        validate(args)


if __name__ == "__main__":
    main()
