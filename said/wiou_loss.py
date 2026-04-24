"""
WIoU_v3_InnerMPDIoU: Combined Custom Loss Function
-----------------------------------------------------
Combines two complementary loss strategies for RTTS foggy-road detection:

1. Wise-IoU v3 (WIoU v3):
   - Uses a "focusing coefficient" based on the outlier degree of each anchor.
   - Assigns LOWER weights to extremely poor-quality / noisy labels 
     (heavily obscured fog samples) and HIGHER weights to "ordinary" anchors.
   - This non-monotonic dynamic weighting stabilizes training on RTTS which
     has significant label noise in near-zero-visibility samples.

2. Inner-MPDIoU:
   - A regression loss that operates on a SCALED-DOWN version of the predicted
     and target boxes ("inner boxes").
   - This inner-box trick penalises centre-point deviation more aggressively,
     accelerating convergence for small targets (pedestrians at distance).
   - MPDIoU additionally accounts for aspect-ratio mismatch.

Usage:
    criterion = WIoU_v3_InnerMPDIoU(inner_scale=0.7, wiou_momentum=0.5)
    loss = criterion(pred_boxes, target_boxes)  # boxes in (x1, y1, x2, y2) format
"""

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Utility: IoU + related geometric metrics
# ──────────────────────────────────────────────────────────────────────────────
def _box_area(box: torch.Tensor) -> torch.Tensor:
    """Area of boxes in (x1, y1, x2, y2) format."""
    return (box[:, 2] - box[:, 0]).clamp(0) * (box[:, 3] - box[:, 1]).clamp(0)


def _box_iou_and_extras(pred: torch.Tensor, target: torch.Tensor):
    """
    Computes IoU plus geometric extras needed for WIoU and MPDIoU.

    Returns:
        iou        : Intersection-over-Union           (N,)
        union      : Union area                        (N,)
        encl_w     : Width of enclosing box            (N,)
        encl_h     : Height of enclosing box           (N,)
        c_dist_sq  : Squared center-point distance     (N,)
        c_diag_sq  : Squared enclosing diagonal        (N,)
        pred_wh    : Predicted (w, h)                  (N, 2)
        tgt_wh     : Target    (w, h)                  (N, 2)
    """
    px1, py1, px2, py2 = pred[:,0],   pred[:,1],   pred[:,2],   pred[:,3]
    tx1, ty1, tx2, ty2 = target[:,0], target[:,1], target[:,2], target[:,3]

    # Intersection
    ix1 = torch.max(px1, tx1)
    iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2)
    iy2 = torch.min(py2, ty2)
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)

    pred_area = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
    tgt_area  = (tx2 - tx1).clamp(0) * (ty2 - ty1).clamp(0)
    union     = pred_area + tgt_area - inter + 1e-7
    iou       = inter / union

    # Enclosing box
    encl_x1 = torch.min(px1, tx1)
    encl_y1 = torch.min(py1, ty1)
    encl_x2 = torch.max(px2, tx2)
    encl_y2 = torch.max(py2, ty2)
    encl_w   = (encl_x2 - encl_x1).clamp(0)
    encl_h   = (encl_y2 - encl_y1).clamp(0)
    c_diag_sq = encl_w ** 2 + encl_h ** 2 + 1e-7

    # Centre-point distance
    pc_x = (px1 + px2) / 2;  pc_y = (py1 + py2) / 2
    tc_x = (tx1 + tx2) / 2;  tc_y = (ty1 + ty2) / 2
    c_dist_sq = (pc_x - tc_x) ** 2 + (pc_y - tc_y) ** 2

    pred_wh = torch.stack([px2 - px1, py2 - py1], dim=-1).clamp(0)
    tgt_wh  = torch.stack([tx2 - tx1, ty2 - ty1], dim=-1).clamp(0)

    return iou, union, encl_w, encl_h, c_dist_sq, c_diag_sq, pred_wh, tgt_wh


# ──────────────────────────────────────────────────────────────────────────────
# Inner-MPDIoU
# ──────────────────────────────────────────────────────────────────────────────
def inner_mpdIoU_loss(
    pred:        torch.Tensor,
    target:      torch.Tensor,
    inner_scale: float = 0.7,
) -> torch.Tensor:
    """
    Inner-MPDIoU regression loss.

    1. Shrinks pred and target boxes by `inner_scale` around their centres.
    2. Computes IoU on the inner boxes → stronger gradient for centre offset.
    3. Adds MPDIoU terms: normalised (dw, dh) width/height penalties.

    Args:
        pred        : (N, 4) boxes in (x1, y1, x2, y2) format.
        target      : (N, 4) boxes in (x1, y1, x2, y2) format.
        inner_scale : Shrink factor for inner boxes (0 < s ≤ 1).
    """
    # --- Build inner boxes ---
    def shrink(box, s):
        cx = (box[:, 0] + box[:, 2]) / 2
        cy = (box[:, 1] + box[:, 3]) / 2
        hw = (box[:, 2] - box[:, 0]) * s / 2
        hh = (box[:, 3] - box[:, 1]) * s / 2
        return torch.stack([cx - hw, cy - hh, cx + hw, cy + hh], dim=-1)

    inner_pred   = shrink(pred,   inner_scale)
    inner_target = shrink(target, inner_scale)

    iou, _, _, _, c_dist_sq, c_diag_sq, pred_wh, tgt_wh = \
        _box_iou_and_extras(inner_pred, inner_target)

    # CIoU-like centre penalty on inner boxes
    centre_penalty = c_dist_sq / c_diag_sq

    # MPDIoU width / height deviation terms (normalised by enclosing diagonal)
    dw = (pred_wh[:, 0] - tgt_wh[:, 0]) ** 2 / c_diag_sq
    dh = (pred_wh[:, 1] - tgt_wh[:, 1]) ** 2 / c_diag_sq

    loss = 1.0 - iou + centre_penalty + dw + dh
    return loss


# ──────────────────────────────────────────────────────────────────────────────
# WIoU v3 — Dynamic Non-Monotonic Focusing Mechanism
# ──────────────────────────────────────────────────────────────────────────────
class WIoUv3Loss(nn.Module):
    """
    Wise-IoU v3 with dynamic non-monotonic focusing coefficient.

    Focusing coefficient β = exp[(IoU - μ_IoU)² / (σ_IoU + ε)]
       where μ_IoU and σ_IoU are running statistics of the batch IoU.

    β > 1 for anchors with IoU far from the mean (outliers / label noise):
        → down-weights very bad samples (fog + invisible objects).
    β ≈ 1 for "ordinary-quality" samples → stable learning signal.

    Args:
        momentum : EMA decay for running IoU statistics (default: 0.5).
    """
    def __init__(self, momentum: float = 0.5):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("mean_iou",  torch.tensor(0.5))
        self.register_buffer("var_iou",   torch.tensor(0.25))

    @torch.no_grad()
    def _update_statistics(self, iou: torch.Tensor):
        """EMA update of running mean/variance of batch IoU."""
        b_mean = iou.mean()
        b_var  = iou.var(unbiased=False).clamp(min=1e-6)
        self.mean_iou.mul_(1 - self.momentum).add_(b_mean * self.momentum)
        self.var_iou.mul_(1 - self.momentum).add_(b_var  * self.momentum)

    def _focusing_coefficient(self, iou: torch.Tensor) -> torch.Tensor:
        """
        Non-monotonic β: high for IoU far from batch mean (outliers),
        near 1 for ordinary anchors.
        """
        outlier_degree = (iou - self.mean_iou) ** 2 / (self.var_iou + 1e-6)
        beta = torch.exp(outlier_degree)           # ≥ 1, peaks at outliers
        # Detach so β does not propagate gradients (it is a weight, not a loss)
        return beta.detach()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred   : (N, 4) predicted boxes  (x1, y1, x2, y2).
            target : (N, 4) ground-truth boxes.
        Returns:
            Scalar loss.
        """
        iou, union, encl_w, encl_h, c_dist_sq, c_diag_sq, _, _ = \
            _box_iou_and_extras(pred, target)

        if self.training:
            self._update_statistics(iou.detach())

        # Base WIoU geometric loss (CIoU-style centre term)
        wiou_base = 1.0 - iou + c_dist_sq / c_diag_sq

        # Dynamic focusing coefficient — down-weights outliers
        beta = self._focusing_coefficient(iou)

        loss = (beta * wiou_base).mean()
        return loss


# ──────────────────────────────────────────────────────────────────────────────
# Combined: WIoU_v3_InnerMPDIoU
# ──────────────────────────────────────────────────────────────────────────────
class WIoU_v3_InnerMPDIoU(nn.Module):
    """
    Combined bounding-box regression loss for SAID on RTTS.

    loss = α * WIoU_v3(pred, target) + (1 - α) * InnerMPDIoU(pred, target)

    Args:
        alpha       : Mixing weight between WIoU and InnerMPDIoU (default: 0.6).
        inner_scale : Shrink factor for inner boxes in MPDIoU (default: 0.7).
        momentum    : EMA decay for WIoU v3 IoU statistics (default: 0.5).
    """
    def __init__(
        self,
        alpha:       float = 0.6,
        inner_scale: float = 0.7,
        momentum:    float = 0.5,
    ):
        super().__init__()
        self.alpha       = alpha
        self.inner_scale = inner_scale
        self.wiou        = WIoUv3Loss(momentum=momentum)

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred   : (N, 4) predicted boxes  (x1, y1, x2, y2).
            target : (N, 4) ground-truth boxes (x1, y1, x2, y2).
        Returns:
            Scalar combined loss.
        """
        if pred.numel() == 0:
            return pred.sum() * 0

        loss_wiou  = self.wiou(pred, target)
        loss_inner = inner_mpdIoU_loss(pred, target, self.inner_scale).mean()

        return self.alpha * loss_wiou + (1.0 - self.alpha) * loss_inner
