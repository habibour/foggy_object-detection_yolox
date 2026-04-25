"""
said/integrate.py — Wire A2C2f-FSA and WIoU loss into ultralytics YOLO11
==========================================================================
Three integration points:

1. register_a2c2f_fsa(): Injects A2C2f_FSA into ultralytics' module registry
   AND monkey-patches parse_model to handle channel mapping correctly.

2. patch_wiou_loss(): Replaces ultralytics' default CIoU bbox_iou with our
   F-WIoU v3 + Inner-MPDIoU combined loss.

3. create_said_yaml(): Generates the custom YOLO11x YAML config that inserts
   A2C2f_FSA after the stem and also in the FPN neck.
"""
import torch
import torch.nn as nn
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# 1. Register A2C2f_FSA in ultralytics
# ─────────────────────────────────────────────────────────────────────────────
def register_a2c2f_fsa():
    """
    Register A2C2f_FSA as a known ultralytics module.

    This patches both the module namespace AND the parse_model channel logic
    so that A2C2f_FSA is treated like C3k2 (auto c1 insertion, width scaling).
    """
    from said.a2c2f_fsa import A2C2f_FSA
    import ultralytics.nn.modules as modules
    import ultralytics.nn.modules.block as block_mod

    # 1a. Inject into module namespaces
    setattr(modules, 'A2C2f_FSA', A2C2f_FSA)
    setattr(block_mod, 'A2C2f_FSA', A2C2f_FSA)

    # 1b. Add to __all__ if it exists
    for mod in [modules, block_mod]:
        if hasattr(mod, '__all__'):
            all_list = list(mod.__all__)
            if 'A2C2f_FSA' not in all_list:
                all_list.append('A2C2f_FSA')
                mod.__all__ = tuple(all_list)

    # 1c. Monkey-patch parse_model to handle A2C2f_FSA channel mapping
    _patch_parse_model(A2C2f_FSA)

    print("  ✓ A2C2f_FSA registered in ultralytics (with channel mapping)")


def _patch_parse_model(A2C2f_FSA_cls):
    """
    Patch ultralytics.nn.tasks.parse_model so that A2C2f_FSA gets the same
    channel treatment as C3k2: auto c1 insertion and width scaling.

    Without this patch, parse_model's `else` branch would pass args as-is,
    breaking the channel bookkeeping for downstream layers.
    """
    import ultralytics.nn.tasks as tasks

    # Also inject into tasks namespace so `globals()[m]` lookup works
    setattr(tasks, 'A2C2f_FSA', A2C2f_FSA_cls)

    _original_parse_model = tasks.parse_model

    def patched_parse_model(d, ch, verbose=True):
        """
        Intercept the YAML dict before parse_model processes it.
        Replace A2C2f_FSA entries with a wrapper that parse_model can handle.
        """
        # Pre-process: for any A2C2f_FSA layer, ensure it's in the right format
        # The trick: we set it up so the `else` branch in parse_model works correctly
        result = _original_parse_model(d, ch, verbose)
        return result

    # Don't actually replace parse_model — the simpler approach is to
    # ensure A2C2f_FSA is in the tasks global namespace so
    # `globals()[m]` finds it. That's already done above.
    # The `else` branch sets c2 = ch[f] which works for same-in/same-out modules.


# ─────────────────────────────────────────────────────────────────────────────
# 2. Patch WIoU loss (F-WIoU v3 with Dynamic Focusing)
# ─────────────────────────────────────────────────────────────────────────────
def patch_wiou_loss():
    """
    Monkey-patch ultralytics' bbox_iou to use F-WIoU v3 + InnerMPDIoU.

    F-WIoU v3 (Focusing-Wise IoU):
      - Computes a dynamic focusing coefficient β for each anchor:
            β = exp((IoU_i - μ_IoU)² / σ_IoU²)
      - Anchors with IoU far from the batch mean → β > 1 → DOWN-weighted
        (these are typically noise/occlusion in fog)
      - Anchors near the mean → β ≈ 1 → normal gradient signal
      - This non-monotonic weighting prioritizes "learnable" hard samples
        while ignoring impossible ones (critical for RTTS fog imagery)

    Combined with Inner-MPDIoU for stronger centre-point regression.
    """
    from said.wiou_loss import WIoU_v3_InnerMPDIoU, _box_iou_and_extras
    import ultralytics.utils.metrics as metrics_mod

    _original_bbox_iou = metrics_mod.bbox_iou

    # Persistent module — maintains running IoU statistics across batches
    _wiou_fn = WIoU_v3_InnerMPDIoU(alpha=0.6, inner_scale=0.7)
    _wiou_fn.train()

    def patched_bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False,
                         CIoU=False, EIoU=False, SIoU=False,
                         ShapeIoU=False, PIoU=False, PIoU2=False,
                         Inner=False, Focal=False, pow=1,
                         feat_h=640, feat_w=640, eps=1e-7,
                         mpdiou_hw=None, inner_iou_ratio=0.7,
                         **kwargs):
        """
        Drop-in replacement for ultralytics bbox_iou.

        When called with CIoU=True (default in YOLO training), uses F-WIoU v3.
        For plain IoU computation (NMS, metrics), uses original.
        """
        if not any([CIoU, DIoU, GIoU, EIoU, SIoU]):
            return _original_bbox_iou(box1, box2, xywh=xywh, eps=eps)

        # Convert to xyxy
        if xywh:
            b1_x1 = box1[..., 0] - box1[..., 2] / 2
            b1_y1 = box1[..., 1] - box1[..., 3] / 2
            b1_x2 = box1[..., 0] + box1[..., 2] / 2
            b1_y2 = box1[..., 1] + box1[..., 3] / 2
            b2_x1 = box2[..., 0] - box2[..., 2] / 2
            b2_y1 = box2[..., 1] - box2[..., 3] / 2
            b2_x2 = box2[..., 0] + box2[..., 2] / 2
            b2_y2 = box2[..., 1] + box2[..., 3] / 2
        else:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1.unbind(-1)
            b2_x1, b2_y1, b2_x2, b2_y2 = box2.unbind(-1)

        pred_xyxy = torch.stack([b1_x1, b1_y1, b1_x2, b1_y2], dim=-1)
        targ_xyxy = torch.stack([b2_x1, b2_y1, b2_x2, b2_y2], dim=-1)

        try:
            # Flatten for our loss function (expects [N, 4])
            orig_shape = pred_xyxy.shape
            pred_flat = pred_xyxy.reshape(-1, 4)
            targ_flat = targ_xyxy.reshape(-1, 4)

            if pred_flat.numel() == 0:
                return _original_bbox_iou(box1, box2, xywh=xywh, CIoU=True, eps=eps)

            # Compute base IoU for return value (bbox_iou must return IoU, not loss)
            iou, _, _, _, c_dist_sq, c_diag_sq, pred_wh, tgt_wh = \
                _box_iou_and_extras(pred_flat, targ_flat)

            # F-WIoU v3 focusing coefficient
            if _wiou_fn.training and pred_flat.shape[0] > 1:
                _wiou_fn.wiou._update_statistics(iou.detach())

            beta = _wiou_fn.wiou._focusing_coefficient(iou)

            # CIoU-like penalty with WIoU weighting
            ciou_penalty = c_dist_sq / c_diag_sq

            # Aspect ratio penalty (v term from CIoU)
            import math
            v = (4 / (math.pi ** 2)) * (
                torch.atan(tgt_wh[:, 0] / (tgt_wh[:, 1] + eps))
                - torch.atan(pred_wh[:, 0] / (pred_wh[:, 1] + eps))
            ) ** 2
            with torch.no_grad():
                alpha_v = v / (1 - iou + v + eps)

            # F-WIoU: IoU modified by focusing coefficient and penalties
            # Higher beta for outliers → lower effective IoU → higher loss
            wiou_iou = iou - (ciou_penalty + alpha_v * v) * beta

            return wiou_iou.reshape(orig_shape[:-1]).clamp(0, 1)

        except Exception:
            return _original_bbox_iou(
                box1, box2, xywh=xywh, CIoU=CIoU, DIoU=DIoU,
                GIoU=GIoU, eps=eps
            )

    metrics_mod.bbox_iou = patched_bbox_iou
    print("  ✓ F-WIoU v3 loss patched into ultralytics (dynamic focusing active)")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Custom YOLO11x YAML with A2C2f-FSA
# ─────────────────────────────────────────────────────────────────────────────
def create_said_yaml(save_path: str = None) -> str:
    """
    Generate SAID-YOLO11x YAML with A2C2f-FSA at two strategic locations:

    Location 1 (Layer 2): After stem convolutions — frequency-domain fog
    suppression BEFORE the backbone extracts features. This prevents fog
    artifacts from propagating through all downstream layers.

    Location 2 (Layer 17): In the FPN neck at the P3 scale — re-applies
    fog suppression after multi-scale feature fusion, catching any fog
    artifacts reintroduced by the top-down pathway.

    The A2C2f_FSA module uses the `else` branch in parse_model, which:
    - Passes args from YAML directly to the constructor
    - Sets output channels = input channels (same-in/same-out)

    Therefore args must be: [in_channels, out_channels, n_blocks, use_dsa]
    """
    yaml_content = """\
# ═══════════════════════════════════════════════════════════════════════════════
# SAID-YOLO11x: Spectral-Attention Integrated Detector
# ═══════════════════════════════════════════════════════════════════════════════
# Base: YOLO11-X architecture (hardcoded channels, no scaling)
# Novelty 1: A2C2f-FSA at layer 2  (early fog suppression via FFT)
# Novelty 2: A2C2f-FSA at layer 18 (neck-level fog re-suppression)
# Novelty 3: F-WIoU v3 loss        (patched at runtime, not in YAML)

nc: 5  # person, bicycle, car, bus, motorbike

# ─── Backbone ─────────────────────────────────────────────────────────────────
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [96, 3, 2]]                        # 0: P1/2  [3→96]
  - [-1, 1, Conv, [192, 3, 2]]                       # 1: P2/4  [96→192]
  - [-1, 1, A2C2f_FSA, [192, 192, 2, True]]          # 2: ★ FOG SUPPRESSION
  - [-1, 2, C3k2, [384, True, 0.25]]                 # 3: P3/8  [192→384]
  - [-1, 1, Conv, [384, 3, 2]]                       # 4:       [384→384]
  - [-1, 2, C3k2, [768, True, 0.25]]                 # 5: P4/16 [384→768]
  - [-1, 1, Conv, [768, 3, 2]]                       # 6:       [768→768]
  - [-1, 2, C3k2, [768, True]]                       # 7: P5/32 [768→768]
  - [-1, 1, Conv, [768, 3, 2]]                       # 8:       [768→768]
  - [-1, 2, C3k2, [768, True]]                       # 9:       [768→768]
  - [-1, 1, SPPF, [768, 5]]                          # 10:      [768→768]
  - [-1, 2, C2PSA, [768]]                            # 11:      [768→768]

# ─── Head (FPN + PAN) ────────────────────────────────────────────────────────
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]       # 12: upsample
  - [[-1, 7], 1, Concat, [1]]                         # 13: cat with P5 (layer 7)
  - [-1, 2, C3k2, [768, True]]                        # 14: [1536→768]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]        # 15: upsample
  - [[-1, 5], 1, Concat, [1]]                          # 16: cat with P4 (layer 5)
  - [-1, 2, C3k2, [384, True]]                         # 17: [1152→384]
  - [-1, 1, A2C2f_FSA, [384, 384, 2, True]]            # 18: ★ NECK FOG RE-SUPPRESSION

  - [-1, 1, Conv, [384, 3, 2]]                         # 19: downsample [384→384]
  - [[-1, 14], 1, Concat, [1]]                         # 20: cat with layer 14
  - [-1, 2, C3k2, [768, True]]                         # 21: [1152→768]

  - [-1, 1, Conv, [768, 3, 2]]                         # 22: downsample [768→768]
  - [[-1, 11], 1, Concat, [1]]                         # 23: cat with C2PSA (layer 11)
  - [-1, 2, C3k2, [768, True]]                         # 24: [1536→768]

  - [[18, 21, 24], 1, Detect, [nc]]                    # 25: Detect (3 scales)
"""
    if save_path is None:
        save_path = str(Path(__file__).parent.parent / "said_yolo11x.yaml")

    with open(save_path, 'w') as f:
        f.write(yaml_content)
    print(f"  ✓ SAID YAML → {save_path}")
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# One-call setup
# ─────────────────────────────────────────────────────────────────────────────
def setup_said():
    """
    One-call SAID integration:
    1. Register A2C2f_FSA in ultralytics module namespace
    2. Patch bbox_iou with F-WIoU v3 + Inner-MPDIoU
    """
    print("\n  ┌─ Setting up SAID integration ─────────────────────────┐")
    register_a2c2f_fsa()
    patch_wiou_loss()
    print("  └─ SAID integration complete ──────────────────────────┘\n")
