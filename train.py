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
    """Return (ROOT, RTTS_YAML, VOC_YAML, WEIGHTS_DIR, CKPT_DIR)."""
    if kaggle or IS_KAGGLE:
        ROOT = Path("/kaggle/working")
    else:
        ROOT = Path(__file__).parent

    RTTS_YAML   = ROOT / "rtts.yaml"
    VOC_YAML    = ROOT / "voc.yaml"       # clean VOC (5 classes)
    WEIGHTS_DIR = ROOT / "weights"
    CKPT_DIR    = ROOT / "checkpoints"

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    return ROOT, RTTS_YAML, VOC_YAML, WEIGHTS_DIR, CKPT_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
def build_common_args(batch: int, workers: int = 4) -> dict:
    """
    Shared training hyperparameters with FOG-AWARE augmentation.

    Fog-Aware Augmentation Strategy:
      Instead of CLAHE preprocessing (adds latency, not GPU-native), we
      simulate contrast enhancement via high hsv_v (0.6) and hsv_s (0.8).
      This forces the model to learn from images with varied contrast/
      saturation, mimicking what CLAHE would produce.

    Memory: Physical batch=4, nbs=64 → 16x grad accumulation → effective 64.
    """
    return dict(
        imgsz         = 640,
        optimizer     = "AdamW",
        lr0           = 0.001,
        lrf           = 0.01,
        weight_decay  = 0.0005,
        warmup_epochs = 3.0,
        warmup_bias_lr= 0.01,
        nbs           = 64,
        # ── Fog-Aware Augmentation ─────────────────────────────────────
        # High hsv_v (0.6): simulates CLAHE — forces varied brightness
        # High hsv_s (0.8): varied saturation combats fog desaturation
        # High scale (0.9): multi-scale training for distance-fog objects
        # erasing (0.3): simulates partial occlusion by dense fog patches
        hsv_h         = 0.015,
        hsv_s         = 0.8,         # high: combats fog desaturation
        hsv_v         = 0.6,         # high: simulates CLAHE contrast var
        flipud        = 0.0,
        fliplr        = 0.5,
        mosaic        = 1.0,
        mixup         = 0.15,
        copy_paste    = 0.1,
        erasing       = 0.3,         # random erasing → fog patch simulation
        degrees       = 0.0,
        translate     = 0.2,         # higher translate for position variety
        scale         = 0.9,         # aggressive multi-scale for fog depth
        shear         = 0.0,
        perspective   = 0.0,
        close_mosaic  = 10,
        amp           = True,
        workers       = workers,
        plots         = True,
        verbose       = True,
        val           = True,
        save          = True,
        save_period   = 10,
    )


def build_phase2b_args(batch: int, workers: int = 4) -> dict:
    """
    Phase 2b args: full fine-tune with cosine LR for smooth convergence.
    cos_lr=True gives better final mAP than linear LR decay.
    """
    base = build_common_args(batch, workers)
    base.update(
        lr0           = 0.001,
        lrf           = 0.001,       # cosine decays to lr0 * lrf = 1e-6
        cos_lr        = True,        # cosine annealing schedule
        warmup_epochs = 3.0,
        close_mosaic  = 5,           # disable augmentation earlier for fine-tune
    )
    return base


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
def make_stability_callbacks(run_dir: Path, root: Path, target_map50: float = 0.93):
    """
    Callbacks for:
    1. Gradient clipping (max_norm=10) after each batch
    2. Forced epoch-1 diagnostic checkpoint
    3. Custom fitness function: 0.9*mAP50 + 0.1*mAP50-95
    4. Early stopping at target_map50 (default 93%)
    """
    import torch

    fitness_state = {
        "best_fitness": 0.0,
        "best_epoch": 0,
        "patience_counter": 0,
        "target_reached": False,
    }

    def on_train_batch_end(trainer):
        """Apply gradient clipping after each batch to prevent NaN gradients."""
        if trainer.model is not None:
            torch.nn.utils.clip_grad_norm_(
                trainer.model.parameters(), max_norm=10.0
            )

    def on_fit_epoch_end_stability(trainer):
        """Epoch-1 save + custom fitness + early stopping."""
        epoch = trainer.epoch + 1

        # ── Force save epoch 1 diagnostic ────────────────────────────────
        if epoch == 1:
            save_dir = Path(trainer.save_dir) / "weights"
            save_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = save_dir / "epoch1_diagnostic.pt"
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': trainer.model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict() if trainer.optimizer else None,
                }, str(ckpt_path))
                print(f"  \U0001f4be Epoch 1 diagnostic saved: {ckpt_path.name}")
                publish_weights(ckpt_path, "said_epoch1_diagnostic.pt", root)
            except Exception as e:
                print(f"  \u26a0 Epoch 1 save failed: {e}")

        # ── Custom fitness: 0.9*mAP50 + 0.1*mAP50-95 ────────────────────
        if hasattr(trainer, 'metrics') and trainer.metrics:
            try:
                map50 = trainer.metrics.get('metrics/mAP50(B)', 0)
                map50_95 = trainer.metrics.get('metrics/mAP50-95(B)', 0)
                fitness = 0.9 * map50 + 0.1 * map50_95

                if fitness > 0:
                    print(
                        f"  \U0001f4ca Fitness={fitness:.4f} "
                        f"(mAP50={map50:.4f}, mAP50-95={map50_95:.4f}) "
                        f"| best={fitness_state['best_fitness']:.4f}"
                    )

                    if fitness > fitness_state["best_fitness"]:
                        fitness_state["best_fitness"] = fitness
                        fitness_state["best_epoch"] = epoch
                        fitness_state["patience_counter"] = 0
                    else:
                        fitness_state["patience_counter"] += 1

                    # ── Early stop: target reached ───────────────────────
                    if map50 >= target_map50:
                        fitness_state["target_reached"] = True
                        print(
                            f"\n  \U0001f3af TARGET REACHED! mAP50={map50:.4f} >= {target_map50} "
                            f"@ epoch {epoch}"
                        )
                        print(f"  Saving best weights and stopping...")
                        # Force save
                        last_pt = Path(trainer.save_dir) / "weights" / "last.pt"
                        if last_pt.exists():
                            publish_weights(last_pt, "said_target_reached.pt", root)
                        trainer.stop = True  # signal ultralytics to stop

                    # ── Early stop: patience exhausted ───────────────────
                    if fitness_state["patience_counter"] >= 10:
                        print(
                            f"\n  \u23f9 EARLY STOP: no improvement for 10 epochs "
                            f"(best fitness={fitness_state['best_fitness']:.4f} "
                            f"@ epoch {fitness_state['best_epoch']})"
                        )
                        trainer.stop = True

            except Exception:
                pass  # metrics not available yet

    return {
        "on_train_batch_end": on_train_batch_end,
        "on_fit_epoch_end": on_fit_epoch_end_stability,
    }


def make_callbacks(args, run_dir: Path, rtts_yaml: str, voc_yaml: str, root: Path):
    """
    Custom eval + checkpoint callbacks for Stage 2.

    Schedule:
      - Val  every args.val_freq  epochs → RTTS val
      - Test every args.test_freq epochs → RTTS test + VOC test
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

        # ── Test every test_freq epochs (RTTS + VOC) ───────────────────────
        if epoch % args.test_freq == 0:
            print(f"\n{'─'*55}\n TEST @ epoch {epoch}\n{'─'*55}")
            _run_val("TEST-RTTS",   epoch, "test", rtts_yaml)
            if Path(voc_yaml).exists():
                _run_val("TEST-VOC", epoch, "test", voc_yaml)

    def on_train_end(trainer):
        epoch = trainer.epoch + 1
        print(f"\n{'═'*55}\n TRAINING COMPLETE\n{'═'*55}")

        # Final evaluation
        print("\n Final RTTS val:")
        _run_val("FINAL-VAL-RTTS",    epoch, "val",  rtts_yaml)
        print("\n Final RTTS test:")
        _run_val("FINAL-TEST-RTTS",   epoch, "test", rtts_yaml)
        if Path(voc_yaml).exists():
            print("\n Final VOC test:")
            _run_val("FINAL-TEST-VOC", epoch, "test", voc_yaml)

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
# Stage 1: Pre-train on CLEAN VOC (like the paper: VOC07+12)
# ─────────────────────────────────────────────────────────────────────────────
def stage1_voc(args, paths):
    """
    Pre-train SAID model on clean VOC images (5 classes).

    Rationale (matching the paper):
      - VOC has clear, high-quality images → teaches general object detection
      - Same 5 classes as RTTS: person, bicycle, car, bus, motorbike
      - No fog-specific augmentation needed (clean images)
      - A2C2f-FSA learns neutral feature extraction on clean data
      - Then fine-tuning on RTTS activates fog suppression
    """
    ROOT, RTTS_YAML, VOC_YAML, WEIGHTS_DIR, CKPT_DIR = paths
    STAGE1_WEIGHTS = WEIGHTS_DIR / "said_voc_pretrained.pt"

    if not VOC_YAML.exists():
        raise FileNotFoundError(
            f"VOC config not found: {VOC_YAML}\n"
            f"Create voc.yaml with paths to your clean VOC dataset."
        )

    print("\n" + "=" * 55)
    print(" SAID — Stage 1: Pre-training on CLEAN VOC")
    print(" (Matching paper: learn general detection on clear images)")
    print("=" * 55)

    # Build SAID model: YOLO11x + A2C2f-FSA
    from said.integrate import create_said_yaml, register_a2c2f_fsa
    register_a2c2f_fsa()
    yaml_path = create_said_yaml(str(ROOT / "said_yolo11x.yaml"))
    model = YOLO(yaml_path)
    print(f"  SAID model: {sum(p.numel() for p in model.model.parameters()):,} params")

    # Attach stability callbacks
    stab_cbs = make_stability_callbacks(
        run_dir=ROOT / "runs" / "said" / "stage1_voc", root=ROOT,
        target_map50=1.0,  # don't early-stop Stage 1
    )
    for event, fn in stab_cbs.items():
        model.add_callback(event, fn)

    # Clean-image augmentation: standard, no fog-specific tricks
    results = model.train(
        data         = str(VOC_YAML),
        epochs       = args.s1_epochs,
        batch        = args.batch,
        device       = args.device,
        project      = str(ROOT / "runs" / "said"),
        name         = "stage1_voc",
        exist_ok     = True,
        patience     = 15,
        # Standard hyperparameters (no fog augmentation)
        imgsz        = 640,
        optimizer    = "AdamW",
        lr0          = 0.002,        # higher LR OK for clean images
        lrf          = 0.01,
        weight_decay = 0.0005,
        warmup_epochs= 3.0,
        warmup_bias_lr= 0.01,
        nbs          = 64,
        # Standard augmentation (clean images — no fog simulation)
        hsv_h        = 0.015,
        hsv_s        = 0.7,
        hsv_v        = 0.4,          # normal brightness variation
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
        amp          = True,
        workers      = 4,
        plots        = True,
        verbose      = True,
        val          = True,
        save         = True,
        save_period  = 10,
    )

    # Publish Stage 1 weights
    best = Path(results.save_dir) / "weights" / "best.pt"
    shutil.copy2(best, STAGE1_WEIGHTS)
    publish_weights(best, "said_stage1_voc_best.pt", ROOT)
    print(f"\nStage 1 complete → {STAGE1_WEIGHTS}")
    return str(STAGE1_WEIGHTS)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Fine-tune on RTTS
# ─────────────────────────────────────────────────────────────────────────────
def stage2_rtts(args, paths, init_weights: str = None):
    ROOT, RTTS_YAML, VOC_YAML, WEIGHTS_DIR, CKPT_DIR = paths
    STAGE1_WEIGHTS = WEIGHTS_DIR / "said_voc_pretrained.pt"

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
            from said.integrate import create_said_yaml, register_a2c2f_fsa
            register_a2c2f_fsa()
            init_weights = create_said_yaml(str(ROOT / "said_yolo11x.yaml"))
    print(f"\nInitialising Stage 2 from: {init_weights}")

    print("\n" + "=" * 55)
    print(" SAID — Stage 2: Fine-tuning on RTTS")
    print("=" * 55)

    # ── Phase 2a: Freeze backbone (15 epochs) ─────────────────────────────
    # Freezes layers 0-9 (backbone) to stabilize detection heads first.
    # Uses higher LR for unfrozen head layers with linear schedule.
    freeze_epochs = 15
    print(f"\n[Phase 2a] Backbone frozen (layers 0-9) — {freeze_epochs} epochs")
    model_a = YOLO(init_weights)

    # Attach stability callbacks to Phase 2a too
    stab_2a = make_stability_callbacks(
        run_dir=ROOT / "runs" / "said" / "stage2a_freeze", root=ROOT,
        target_map50=args.target_map,
    )
    for event, fn in stab_2a.items():
        model_a.add_callback(event, fn)

    model_a.train(
        data         = str(RTTS_YAML),
        epochs       = freeze_epochs,
        batch        = args.batch,
        device       = args.device,
        project      = str(ROOT / "runs" / "said"),
        name         = "stage2a_freeze",
        exist_ok     = True,
        freeze       = list(range(10)),   # freeze backbone layers 0-9
        lr0          = 0.005,             # higher LR for head-only training
        lrf          = 0.1,
        weight_decay = 0.0005,
        warmup_epochs= 2.0,
        nbs          = 64,
        imgsz        = 640,
        optimizer    = "AdamW",
        # Fog-aware augmentation (lighter for warmup)
        hsv_h        = 0.015,
        hsv_s        = 0.7,
        hsv_v        = 0.5,              # moderate CLAHE simulation
        mosaic       = 0.8,
        mixup        = 0.1,
        erasing      = 0.2,
        scale        = 0.5,
        amp          = True,
        workers      = 4,
        val          = True,
        save         = True,
        save_period  = 10,
        plots        = True,
        verbose      = True,
    )

    # Find phase2a best weights
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

    # ── Phase 2b: Full fine-tune with cosine LR ───────────────────────────
    # Unfreezes all layers, uses cos_lr for smooth convergence.
    # Custom fitness (0.9*mAP50 + 0.1*mAP50-95) with patience=10.
    remaining_epochs = max(args.epochs - freeze_epochs, 35)
    print(f"\n[Phase 2b] Full fine-tune with cos_lr — {remaining_epochs} epochs")
    print(f"  Fitness: 0.9*mAP50 + 0.1*mAP50-95")
    print(f"  Early stop: patience=10 or mAP50 >= {args.target_map}")

    run_name = "stage2b_full"
    run_dir  = ROOT / "runs" / "said" / run_name

    phase2b_args = build_phase2b_args(args.batch)
    model_b = YOLO(str(phase2a_best))

    # Eval + checkpoint callbacks
    callbacks = make_callbacks(
        args=args, run_dir=run_dir,
        rtts_yaml=str(RTTS_YAML), voc_yaml=str(VOC_YAML),
        root=ROOT,
    )
    for event, fn in callbacks.items():
        model_b.add_callback(event, fn)

    # Stability + fitness + early stopping callbacks
    stab_cbs = make_stability_callbacks(
        run_dir=run_dir, root=ROOT, target_map50=args.target_map,
    )
    for event, fn in stab_cbs.items():
        model_b.add_callback(event, fn)

    model_b.train(
        data     = str(RTTS_YAML),
        epochs   = remaining_epochs,
        batch    = args.batch,
        device   = args.device,
        project  = str(ROOT / "runs" / "said"),
        name     = run_name,
        exist_ok = True,
        patience = 0,            # we handle early stopping in our callback
        **{**phase2b_args, "val": True},  # enable val for fitness tracking
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
    p.add_argument("--stage",     choices=["voc","rtts","both","validate","check"],
                   default="both")
    p.add_argument("--epochs",    type=int, default=100,
                   help="Stage 2 total epochs (Phase 2a + 2b)")
    p.add_argument("--s1-epochs", type=int, default=50,
                   help="Stage 1 clean VOC pre-training epochs")
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
                   help="Test every N epochs on RTTS+VOC (default: 20)")
    p.add_argument("--save-freq", type=int, default=5,
                   help="Save rolling best every N epochs (default: 5)")
    p.add_argument("--target-map", type=float, default=0.93,
                   help="Target mAP50 for early stopping (default: 0.93)")
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
    print(f"  Target mAP50: {args.target_map} (early stop)")
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
    if args.stage in ("voc", "both"):
        stage1_weights = stage1_voc(args, paths)

    if args.stage in ("rtts", "both"):
        stage2_rtts(args, paths, init_weights=stage1_weights)

    if args.stage in ("rtts", "both"):
        validate(args, paths)


if __name__ == "__main__":
    main()
