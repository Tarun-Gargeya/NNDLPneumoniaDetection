"""Data loading, augmentation, model and Grad-CAM utilities."""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
INPUT = 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
SHORTCUT_COLS = ["log_area", "aspect"]

VARIANTS = {
    # name: (cam_loss, acquisition_aug, adversary)
    "baseline": (False, False, False),
    "camguide": (True, False, False),
    "camguide_aug": (True, True, False),
    "ours": (True, True, True),
}


def load_data():
    df = pd.read_csv(PROC / "index.csv")
    images = np.load(PROC / "images.npy", mmap_mode="r")
    masks = np.load(PROC / "masks.npy", mmap_mode="r")
    return df, images, masks


def normalize(x: torch.Tensor) -> torch.Tensor:
    """(B,1,H,W) in [0,1] -> (B,3,H,W) with ImageNet normalization."""
    x = x.expand(-1, 3, -1, -1)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def eval_resize(x: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, size=INPUT, mode="bilinear", align_corners=False, antialias=True)


def augment(x: torch.Tensor, m: torch.Tensor, acq: bool):
    """Batched augmentation of images and their lung masks.

    Always: flip, rotation +-15 deg, zoom 0.9-1.1, brightness jitter.
    acq=True: also aspect-ratio stretch, gamma, contrast, downsampling and noise.
    """
    b, dev = x.size(0), x.device
    ang = (torch.rand(b, device=dev) * 30 - 15) * math.pi / 180
    scale = torch.rand(b, device=dev) * 0.2 + 0.9
    flip = (torch.rand(b, device=dev) < 0.5).float() * 2 - 1
    cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
    zero = torch.zeros_like(cos)
    ar = torch.exp((torch.rand(b, device=dev) - 0.5) * 0.6) if acq else torch.ones_like(cos)
    theta = torch.stack([torch.stack([cos * flip * ar, -sin, zero], 1),
                         torch.stack([sin * flip * ar, cos, zero], 1)], 1)
    grid = F.affine_grid(theta, (b, 1, INPUT, INPUT), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    m = (F.grid_sample(m, grid, mode="bilinear", align_corners=False) > 0.5).float()
    x = x + (torch.rand(b, 1, 1, 1, device=dev) - 0.5) * 0.2

    if acq:
        x = x.clamp(1e-4, 1)
        gamma = torch.exp((torch.rand(b, 1, 1, 1, device=dev) - 0.5) * 1.0)  # ~0.6-1.65
        x = x ** gamma
        mu = x.mean((2, 3), keepdim=True)
        x = (x - mu) * (0.7 + 0.6 * torch.rand(b, 1, 1, 1, device=dev)) + mu
        x = x + (torch.rand(b, 1, 1, 1, device=dev) - 0.5) * 0.3
        low = int(torch.randint(64, INPUT + 1, (1,)))
        x = F.interpolate(F.interpolate(x, size=low, mode="bilinear", antialias=True),
                          size=INPUT, mode="bilinear")
        x = x + torch.randn_like(x) * 0.03 * torch.rand(b, 1, 1, 1, device=dev)
    return x.clamp(0, 1), m


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.coeff * g, None


class PneumoniaNet(nn.Module):
    """EfficientNet-B3 -> GAP -> Dense(256) -> Dropout(0.5) -> 2 logits.

    With adversary=True, an extra head predicts the image's original size and aspect
    ratio through a gradient reversal layer.
    """

    def __init__(self, pretrained=True, adversary=False):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b3", pretrained=pretrained, num_classes=0)
        c = self.backbone.num_features
        self.head = nn.Sequential(nn.Linear(c, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 2))
        self.adversary = (nn.Sequential(nn.Linear(c, 256), nn.ReLU(), nn.Linear(256, len(SHORTCUT_COLS)))
                          if adversary else None)

    def forward(self, x, grl=1.0):
        fmap = self.backbone.forward_features(x)  # (B, 1536, 7, 7)
        if not fmap.requires_grad and torch.is_grad_enabled():
            fmap.requires_grad_()  # needed for Grad-CAM when the backbone is frozen
        pooled = fmap.mean((2, 3))
        logits = self.head(pooled)
        adv = self.adversary(GradReverse.apply(pooled, grl)) if self.adversary is not None else None
        return logits, fmap, pooled, adv

    def fine_tune_params(self):
        """Parameters unfrozen in phase 2: last two stages and the head conv."""
        bb = self.backbone
        return [*bb.blocks[-2:].parameters(), *bb.conv_head.parameters(), *bb.bn2.parameters()]


def grad_cam(logits, fmap, target, create_graph):
    """Grad-CAM maps (B,h,w) for the target class. Differentiable if create_graph=True."""
    score = logits.gather(1, target[:, None]).sum()
    g, = torch.autograd.grad(score, fmap, create_graph=create_graph, retain_graph=True)
    w = g.mean((2, 3), keepdim=True)
    return F.relu((w * fmap).sum(1))


def outside_fraction(cam, mask):
    """Fraction of CAM mass outside the lung mask."""
    m = F.adaptive_avg_pool2d(mask, cam.shape[-2:])[:, 0]
    return (cam * (1 - m)).sum((1, 2)) / (cam.sum((1, 2)) + 1e-6)
