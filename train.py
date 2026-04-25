"""
SAID Training Script — Two-Stage Fine-Tuning via Ultralytics API
=================================================================
Stage 1: Pre-train on VOC-FOG  (synthetic foggy, 9,578 train images)
Stage 2: Fine-tune on RTTS     (real-world foggy, 4,322 images, 60/20/20)

Architecture highlights:
  - Base: YOLO11-X pretrained weights (yolo11x.pt)
  - Preprocessing  → A2C2f-FSA (FFT frequency disentanglement + DSA)
  - Loss           → WIoU_v3_InnerMPDIoU (difficulty-adaptive weighting)

Checkpoint & Evaluation Schedule:
  - Both stages  : checkpoint saved every 10 epochs + best.pt + last.pt
  - Stage 2 val  : every 10 epochs on RTTS val
  - Stage 2 test : every 20 epochs on RTTS test + VOC-FOG test
  - Rolling best : every 5 epochs (replaces if mAP improved)
  - All .pt files saved to /kaggle/working/ for easy download

Resume training:
    python train.py --stage rtts --resume /path/to/last.pt --kaggle

Usage:
    python train.py --stage both --kaggle
    python train.py --stage rtts --epochs 100 --kaggle
    python train.py --stage validate --kaggle
"""

import argparse
import csv
import glob
import json
import os
import shutil
from pathlib import Path

from ultralytics import YOLO

# ── CUDA memory optimization (must be set BEFORE any torch.cuda calls) ────────
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Import SAID integration (registers A2C2f-FSA + patches WIoU loss)
import sys
if '/kaggle/working' in sys.path or os.path.exists('/kaggle/working'):
    sys.path.insert(0, '/kaggle/working')



# ─────────────────────────────────────────────────────────────────────────────
# Environment Detection
# ─────────────────────────────────────────────────────────────────────────────
IS_KAGGLE = os.path.exists("/kaggle/working")


def auto_device() -> str:
    """Return the best available device string."""
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def get_paths(kaggle: bool = False):
    """Return (ROOT, RTTS_YAML, VOCFOG_YAML, WEIGHTS_DIR, CKPT_DIR)."""
    if kaggle or IS_KAGGLE:
        ROOT = Path("/kaggle/working")
    else:
        ROOT = Path(__file__).parent

    RTTS_YAML   = ROOT / "rtts.yaml"
    VOCFOG_YAML = ROOT / "vocfog.yaml"
    WEIGHTS_DIR = ROOT / "weights"
    CKPT_DIR    = ROOT / "checkpoints"

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    return ROOT, RTTS_YAML, VOCFOG_YAML, WEIGHTS_DIR, CKPT_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
def build_common_args(batch: int, workers: int = 4) -> dict:
    """
    Shared training hyperparameters.

    Memory strategy for P100 (16GB):
      - Physical batch = 4 (fits in VRAM with YOLO11x + augmentation)
      - nbs = 64 (nominal batch size → gradient accumulated over 64/4 = 16 steps)
      - This gives effective batch 64 with only 4 images per GPU forward pass

    Stability:
      - lr0 = 0.001 (conservative to prevent EMA NaN/Inf)
      - warmup_epochs = 5.0 (longer warmup for stable gradient scaling)
      - max_norm gradient clipping handled by ultralytics internally
    """
    return dict(
        imgsz         = 640,
        optimizer     = "AdamW",
        lr0           = 0.001,       # conservative LR to prevent EMA NaN
        lrf           = 0.01,
        weight_decay  = 0.0005,
        warmup_epochs = 5.0,         # longer warmup for gradient stability
        warmup_bias_lr= 0.01,        # lower bias warmup LR
        nbs           = 64,          # nominal batch size → grad accum = 64/batch
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
        amp           = True,        # FP16 mixed precision
        workers       = workers,
        plots         = True,
        verbose       = True,
        val           = True,
        save          = True,
        save_period   = 10,          # save checkpoint every 10 epochs
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper: copy checkpoints to /kaggle/working/ for download
# ─────────────────────────────────────────────────────────────────────────────
def publish_weights(src: Path, name: str, root: Path):
    """Copy a .pt file to ROOT (e.g. /kaggle/working/) for easy download."""
    dest = root / name
    if src.exists():
        shutil.copy2(src, dest)
        size_mb = dest.stat().st_size / 1e6
        print(f"  📦 {name} ({size_mb:.1f} MB) → {dest}")
    return dest


# ─────────────────────────────────────────────────────────────────────────────
# Callback Factory for Stage 2
# ─────────────────────────────────────────────────────────────────────────────
def make_stability_callbacks(run_dir: Path, root: Path):
    """
    Callbacks for gradient clipping and forced first-epoch checkpoint.
    Prevents EMA NaN/Inf and ensures diagnostic checkpoint is always available.
    """
    import torch

    def on_train_batch_end(trainer):
        """Apply gradient clipping after each batch to prevent NaN gradients."""
        if trainer.model is not None and trainer.model.parameters():
            torch.nn.utils.clip_grad_norm_(
                trainer.model.parameters(), max_norm=10.0
            )

    def on_fit_epoch_end_stability(trainer):
        """Force-save epoch 1 checkpoint regardless of EMA status."""
        epoch = trainer.epoch + 1
        if epoch == 1:
            save_dir = Path(trainer.save_dir) / "weights"
            save_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = save_dir / "epoch1_diagnostic.pt"
            try:
                # Save model state dict directly (bypasses EMA check)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': trainer.model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict() if trainer.optimizer else None,
                }, str(ckpt_path))
                print(f"  💾 Epoch 1 diagnostic checkpoint saved: {ckpt_path.name}")
                publish_weights(ckpt_path, "said_epoch1_diagnostic.pt", root)
            except Exception as e:
                print(f"  ⚠ Epoch 1 save failed: {e}")

    return {
        "on_train_batch_end": on_train_batch_end,
        "on_fit_epoch_end": on_fit_epoch_end_stability,
    }


def make_callbacks(args, run_dir: Path, rtts_yaml: str, vocfog_yaml: str, root: Path):
    """
    Custom eval + checkpoint callbacks for Stage 2.

    Schedule:
      - Val  every args.val_freq  epochs → RTTS val
      - Test every args.test_freq epochs → RTTS test + VOC-FOG test
      - Save rolling best every args.save_freq epochs (if mAP improved)
      - Publish .pt files to ROOT for Kaggle download
    """
    WEIGHTS_DIR = root / "weights"
    CKPT_DIR    = root / "checkpoints"
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
                # Also publish to ROOT for download
                publish_weights(last_pt, "said_best.pt", root)
                print(
                    f"  ★ Rolling best updated: mAP50={map50:.4f} "
                    f"@ epoch {epoch}"
                )

    def on_fit_epoch_end(trainer):
        epoch = trainer.epoch + 1

        # ── Always publish last.pt for crash recovery ─────────────────────
        last_pt = run_dir / "weights" / "last.pt"
        if last_pt.exists():
            publish_weights(last_pt, "said_last.pt", root)

        # ── Publish periodic checkpoints (epoch_N.pt) every 10 epochs ─────
        if epoch % 10 == 0:
            epoch_pt = run_dir / "weights" / f"epoch{epoch}.pt"
            if epoch_pt.exists():
                publish_weights(epoch_pt, f"said_epoch{epoch}.pt", root)
            # Also copy last.pt as a named checkpoint
            if last_pt.exists():
                publish_weights(last_pt, f"said_epoch{epoch}.pt", root)

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

        # Publish final weights
        best_pt = run_dir / "weights" / "best.pt"
        last_pt = run_dir / "weights" / "last.pt"
        if best_pt.exists():
            shutil.copy2(best_pt, STAGE2_WEIGHTS)
            publish_weights(best_pt, "said_final_best.pt", root)
        if last_pt.exists():
            publish_weights(last_pt, "said_final_last.pt", root)

        # Summary JSON
        summary = {
            "best_rolling_map50": state["best_map50"],
            "best_rolling_epoch": state["best_epoch"],
            "rolling_best":       str(ROLLING_BEST),
            "final_weights":      str(STAGE2_WEIGHTS),
            "eval_log":           str(LOG_PATH),
        }
        summary_path = root / "said_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"\n  Rolling best: mAP50={state['best_map50']:.4f} @ ep{state['best_epoch']}")
        print(f"  Eval log    : {LOG_PATH}")
        print(f"  Summary     : {summary_path}")

        # List all downloadable files
        print(f"\n{'─'*55}\n Downloadable files in {root}:\n{'─'*55}")
        for f in sorted(root.glob("said_*.pt")):
            print(f"  📦 {f.name}  ({f.stat().st_size/1e6:.1f} MB)")

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

    # Build SAID model: YOLO11x + A2C2f-FSA fog suppression
    from said.integrate import create_said_yaml, register_a2c2f_fsa
    register_a2c2f_fsa()
    yaml_path = create_said_yaml(str(ROOT / "said_yolo11x.yaml"))
    model = YOLO(yaml_path)
    print(f"  SAID model: {sum(p.numel() for p in model.model.parameters()):,} params")

    # Attach stability callbacks (gradient clipping + epoch-1 diagnostic save)
    stab_cbs = make_stability_callbacks(
        run_dir=ROOT / "runs" / "said" / "stage1_vocfog", root=ROOT
    )
    for event, fn in stab_cbs.items():
        model.add_callback(event, fn)

    results = model.train(
        data    = str(VOCFOG_YAML),
        epochs  = args.s1_epochs,
        batch   = args.batch,
        device  = args.device,
        project = str(ROOT / "runs" / "said"),
        name    = "stage1_vocfog",
        exist_ok= True,
        patience= 15,
        **common,
    )

    # Publish Stage 1 weights
    best = Path(results.save_dir) / "weights" / "best.pt"
    shutil.copy2(best, STAGE1_WEIGHTS)
    publish_weights(best, "said_stage1_best.pt", ROOT)
    print(f"\nStage 1 complete → {STAGE1_WEIGHTS}")
    return str(STAGE1_WEIGHTS)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Fine-tune on RTTS
# ─────────────────────────────────────────────────────────────────────────────
def stage2_rtts(args, paths, init_weights: str = None):
    ROOT, RTTS_YAML, VOCFOG_YAML, WEIGHTS_DIR, CKPT_DIR = paths
    STAGE1_WEIGHTS = WEIGHTS_DIR / "said_vocfog_pretrained.pt"

    # ── Handle resume ─────────────────────────────────────────────────────
    if args.resume:
        resume_pt = Path(args.resume)
        if not resume_pt.exists():
            # Check in ROOT
            resume_pt = ROOT / args.resume
        if resume_pt.exists():
            print(f"\n🔄 RESUMING training from: {resume_pt}")
            model_b = YOLO(str(resume_pt))

            run_name = "stage2b_full"
            run_dir  = ROOT / "runs" / "said" / run_name
            common   = build_common_args(args.batch)

            callbacks = make_callbacks(
                args=args, run_dir=run_dir,
                rtts_yaml=str(RTTS_YAML), vocfog_yaml=str(VOCFOG_YAML),
                root=ROOT,
            )
            for event, fn in callbacks.items():
                model_b.add_callback(event, fn)

            model_b.train(
                data     = str(RTTS_YAML),
                epochs   = args.epochs,
                batch    = args.batch,
                device   = args.device,
                project  = str(ROOT / "runs" / "said"),
                name     = run_name,
                exist_ok = True,
                patience = 0,
                resume   = True,
                **{**common, "val": False},
            )
            print(f"\nStage 2 (resumed) complete.")
            return
        else:
            print(f"  ⚠ Resume file not found: {args.resume}, starting fresh")

    # ── Normal Stage 2 flow ───────────────────────────────────────────────
    if init_weights is None:
        if STAGE1_WEIGHTS.exists():
            init_weights = str(STAGE1_WEIGHTS)
        else:
            # Build SAID model from scratch if no Stage 1 weights
            from said.integrate import create_said_yaml, register_a2c2f_fsa
            register_a2c2f_fsa()
            init_weights = create_said_yaml(str(ROOT / "said_yolo11x.yaml"))
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
        exist_ok     = True,
        freeze       = list(range(10)),
        lr0          = 0.0005,
        lrf          = 0.1,
        weight_decay = 0.0005,
        warmup_epochs= 2.0,
        nbs          = 64,
        imgsz        = 640,
        optimizer    = "AdamW",
        mosaic       = 0.8,
        amp          = True,
        workers      = 4,
        val          = True,
        save         = True,
        save_period  = 10,
        plots        = True,
        verbose      = True,
    )

    # Ultralytics may append -N suffix (stage2a_freeze-2, etc.)
    phase2a_best = ROOT / "runs" / "said" / "stage2a_freeze" / "weights" / "best.pt"
    if not phase2a_best.exists():
        candidates = sorted(glob.glob(str(ROOT / "runs" / "said" / "stage2a_freeze*" / "weights" / "best.pt")))
        if candidates:
            phase2a_best = Path(candidates[-1])
            print(f"  Found phase2a best at: {phase2a_best}")
        else:
            print(f"  Warning: phase2a best not found, using {init_weights}")
            phase2a_best = Path(init_weights)

    publish_weights(phase2a_best, "said_stage2a_best.pt", ROOT)

    # ── Phase 2b: Full fine-tune with custom callbacks ─────────────────────
    print("\n[Phase 2b] Full fine-tune on RTTS with custom eval schedule")
    run_name = "stage2b_full"
    run_dir  = ROOT / "runs" / "said" / run_name

    common  = build_common_args(args.batch)
    model_b = YOLO(str(phase2a_best))

    callbacks = make_callbacks(
        args=args, run_dir=run_dir,
        rtts_yaml=str(RTTS_YAML), vocfog_yaml=str(VOCFOG_YAML),
        root=ROOT,
    )
    for event, fn in callbacks.items():
        model_b.add_callback(event, fn)

    # Stability callbacks (gradient clipping + epoch-1 save)
    stab_cbs = make_stability_callbacks(run_dir=run_dir, root=ROOT)
    for event, fn in stab_cbs.items():
        model_b.add_callback(event, fn)

    model_b.train(
        data     = str(RTTS_YAML),
        epochs   = args.epochs,
        batch    = args.batch,
        device   = args.device,
        project  = str(ROOT / "runs" / "said"),
        name     = run_name,
        exist_ok = True,
        patience = 0,
        **{**common, "val": False},
    )
    print(f"\nStage 2 complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────
def validate(args, paths):
    ROOT, RTTS_YAML, VOCFOG_YAML, WEIGHTS_DIR, _ = paths
    candidates = [
        args.weights,
        str(WEIGHTS_DIR / "said_rtts_final.pt"),
        str(ROOT / "said_final_best.pt"),
        str(ROOT / "said_best.pt"),
        str(ROOT / "checkpoints" / "said_rolling_best.pt"),
        str(ROOT / "runs" / "said" / "stage2b_full" / "weights" / "best.pt"),
    ]
    weights = None
    for c in candidates:
        if c and Path(c).exists():
            weights = c
            break
    if weights is None:
        print(f"Weights not found. Searched:")
        for c in candidates:
            if c:
                print(f"  ✗ {c}")
        return
    print(f"Using weights: {weights}")

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
                   help="Physical batch size (default: 4, effective 64 via grad accum)")
    p.add_argument("--device",    type=str, default=None,
                   help="Device: '0' (CUDA/P100), 'mps', 'cpu'. Auto-detected.")
    p.add_argument("--weights",   type=str, default=None,
                   help="Weights path for --stage validate")
    p.add_argument("--resume",    type=str, default=None,
                   help="Resume training from a saved .pt checkpoint")
    p.add_argument("--kaggle",    action="store_true",
                   help="Force Kaggle mode")
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

    if args.device is None:
        args.device = auto_device()
    if args.batch is None:
        args.batch = 4  # physical batch; effective = nbs(64) via grad accumulation

    print(f"\n{'═'*55}")
    print(f" SAID Training Configuration")
    print(f"{'═'*55}")
    print(f"  Environment : {'Kaggle P100' if kaggle else 'Local'}")
    print(f"  Device      : {args.device}")
    print(f"  Batch (phys): {args.batch}")
    print(f"  Batch (eff) : 64 (nbs=64, accum={64//max(args.batch,1)} steps)")
    print(f"  Stage       : {args.stage}")
    print(f"  Epochs (S2) : {args.epochs}")
    print(f"  Val every   : {args.val_freq} epochs")
    print(f"  Test every  : {args.test_freq} epochs")
    print(f"  Save best   : every {args.save_freq} epochs")
    if args.resume:
        print(f"  Resume from : {args.resume}")
    print(f"{'═'*55}\n")

    # ── Activate SAID integration: A2C2f-FSA + WIoU loss ──────────────────
    if args.stage != "check":
        try:
            from said.integrate import setup_said
            setup_said()
        except ImportError as e:
            print(f"  ⚠ SAID integration not available: {e}")
            print(f"    Training will use standard YOLO11 architecture + CIoU loss")

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
