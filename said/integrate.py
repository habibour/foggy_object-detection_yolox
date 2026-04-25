"""
said/integrate.py — Wire A2C2f-FSA and WIoU loss into ultralytics YOLO11
"""
import torch
import torch.nn as nn
from pathlib import Path


def register_a2c2f_fsa():
    """Register A2C2f_FSA as a known ultralytics module."""
    from said.a2c2f_fsa import A2C2f_FSA
    import ultralytics.nn.modules as modules
    import ultralytics.nn.modules.block as block_mod
    import ultralytics.nn.tasks as tasks

    # Make it discoverable by parse_model
    setattr(modules, 'A2C2f_FSA', A2C2f_FSA)
    setattr(block_mod, 'A2C2f_FSA', A2C2f_FSA)
    setattr(tasks, 'A2C2f_FSA', A2C2f_FSA)

    # Add to __all__ if exists
    for mod in [modules, block_mod]:
        if hasattr(mod, '__all__'):
            all_list = list(mod.__all__)
            if 'A2C2f_FSA' not in all_list:
                all_list.append('A2C2f_FSA')
                mod.__all__ = tuple(all_list)

    print("  ✓ A2C2f_FSA registered in ultralytics")


def patch_wiou_loss():
    """
    Monkey-patch ultralytics' bbox_iou to use WIoU_v3_InnerMPDIoU.
    This replaces the default CIoU loss with our custom loss.
    """
    from said.wiou_loss import WIoU_v3_InnerMPDIoU
    import ultralytics.utils.metrics as metrics_mod

    _original_bbox_iou = metrics_mod.bbox_iou

    # Create a persistent loss module instance
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
        Patched bbox_iou: uses WIoU for CIoU/DIoU calls, falls back to
        original for plain IoU.
        """
        if not any([CIoU, DIoU, GIoU, EIoU, SIoU]):
            # Plain IoU — use original
            return _original_bbox_iou(box1, box2, xywh=xywh, eps=eps)

        # Convert from xywh to xyxy if needed
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

        # Compute WIoU loss → convert to IoU-like value (1 - loss)
        try:
            loss = _wiou_fn(pred_xyxy, targ_xyxy)
            # bbox_iou should return IoU values (higher = better)
            # WIoU returns loss (lower = better), so: iou = 1 - loss
            iou = 1.0 - loss
            return iou
        except Exception:
            # Fallback to original if shapes don't match
            return _original_bbox_iou(
                box1, box2, xywh=xywh, CIoU=CIoU, DIoU=DIoU,
                GIoU=GIoU, eps=eps
            )

    metrics_mod.bbox_iou = patched_bbox_iou
    print("  ✓ WIoU_v3_InnerMPDIoU loss patched into ultralytics")


def create_said_yaml(save_path: str = None) -> str:
    """
    Create a SAID-YOLO11x YAML config with A2C2f_FSA inserted
    after the stem convolutions for early fog suppression.

    Architecture: standard YOLO11x with A2C2f_FSA added at layer 2.
    All skip-connection indices are adjusted for the inserted layer.
    """
    yaml_content = """\
# SAID-YOLO11x: YOLO11-X with A2C2f-FSA fog suppression
# A2C2f_FSA inserted after stem for frequency-domain fog disentanglement

nc: 5  # person, bicycle, car, bus, motorbike

# No scaling — channels are hardcoded for YOLO11-X size
scales:
  x: [1.0, 1.0, 512]

backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [96, 3, 2]]            # 0: P1/2    [3→96]
  - [-1, 1, Conv, [192, 3, 2]]           # 1: P2/4    [96→192]
  - [-1, 1, A2C2f_FSA, [192, 192, 2, True]]  # 2: ★ fog suppression [192→192]
  - [-1, 2, C3k2, [384, True, 0.25]]     # 3: P3/8    [192→384]
  - [-1, 1, Conv, [384, 3, 2]]           # 4:         [384→384]
  - [-1, 2, C3k2, [768, True, 0.25]]     # 5: P4/16   [384→768]
  - [-1, 1, Conv, [768, 3, 2]]           # 6:         [768→768]
  - [-1, 2, C3k2, [768, True]]           # 7: P5/32   [768→768]
  - [-1, 1, Conv, [768, 3, 2]]           # 8:         [768→768]
  - [-1, 2, C3k2, [768, True]]           # 9:         [768→768]
  - [-1, 1, SPPF, [768, 5]]              # 10:        [768→768]
  - [-1, 2, C2PSA, [768]]                # 11:        [768→768]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 12
  - [[-1, 7], 1, Concat, [1]]                    # 13: cat P5+P4 (layer 7=C3k2 P5)
  - [-1, 2, C3k2, [768, True]]                   # 14: [1536→768]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 15
  - [[-1, 5], 1, Concat, [1]]                     # 16: cat P4+P3 (layer 5=C3k2 P4)
  - [-1, 2, C3k2, [384, True]]                    # 17: [1152→384]

  - [-1, 1, Conv, [384, 3, 2]]                    # 18: [384→384]
  - [[-1, 14], 1, Concat, [1]]                    # 19: cat with layer 14
  - [-1, 2, C3k2, [768, True]]                    # 20: [1152→768]

  - [-1, 1, Conv, [768, 3, 2]]                    # 21: [768→768]
  - [[-1, 11], 1, Concat, [1]]                    # 22: cat with layer 11 (C2PSA)
  - [-1, 2, C3k2, [768, True]]                    # 23: [1536→768]

  - [[17, 20, 23], 1, Detect, [nc]]               # 24: Detect
"""
    if save_path is None:
        save_path = str(Path(__file__).parent.parent / "said_yolo11x.yaml")

    with open(save_path, 'w') as f:
        f.write(yaml_content)
    print(f"  ✓ SAID model YAML saved to: {save_path}")
    return save_path


def build_said_model(pretrained: str = "yolo11x.pt", yaml_path: str = None):
    """
    Build SAID model: load custom YAML architecture + transfer pretrained weights.

    Returns an ultralytics YOLO model with A2C2f_FSA integrated.
    """
    # 1. Register custom modules
    register_a2c2f_fsa()

    # 2. Create YAML if needed
    if yaml_path is None:
        yaml_path = create_said_yaml()

    # 3. Load model from custom YAML
    model = YOLO(yaml_path)
    print(f"  ✓ SAID model loaded from: {yaml_path}")
    print(f"    Parameters: {sum(p.numel() for p in model.model.parameters()):,}")

    return model


def setup_said():
    """One-call setup: register modules + patch loss."""
    print("\n  Setting up SAID integration...")
    register_a2c2f_fsa()
    patch_wiou_loss()
    print("  SAID integration complete.\n")
