"""E10 A11-R0 tiny native-pixel ROI mask refiner + OOF-gated quality head.

Source-only, CPU-synthetic-safe architecture prototype: NOT pretrained, NOT
trained, no checkpoints, no Fold0/TEST and no claim of a new PQ measurement.
The mask-IoU score head MUST only be trained on legitimately out-of-fold
candidate predictions with TRAIN-only annotations; never in-sample A3 outputs.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

INPUT_CHANNELS=3
TRAIN_NATIVE_TILE=256
TRAIN_NATIVE_STRIDE=192
REFINER_OUTPUT_PRIOR_LOGIT=2.0
NEIGHBOR_FORBIDDEN_LOGIT=-20.0
QUALITY_ADVANTAGE_MIN=0.03


def _ensure_source_arrays(image, coarse, neighbors):
    image=np.asarray(image)
    coarse=np.asarray(coarse)
    neighbors=np.asarray(neighbors)
    if image.ndim != 2 or coarse.shape != image.shape or neighbors.shape!=image.shape:
        raise ValueError("Source intensity, original instance and neighbor union must be same 2D geometry")
    if image.shape[0]<16 or image.shape[1]<16:
        raise ValueError("Native ROI tiles must be at least 16 pixels in each dimension")
    if not np.isfinite(image).all():
        raise ValueError("Nonfinite H-alpha pixel intensity")
    for name,mask in (("coarse",coarse),("neighbors",neighbors)):
        if not np.isin(mask,[0,1,False,True]).all():
            raise ValueError(f"{name} is not a binary native-pixel mask")
    if np.any((coarse>0)&(neighbors>0)):
        raise ValueError("Input candidate and neighbors must be disjoint under A6 owner policy")
    return image,coarse,neighbors


def source_roi_channels(image,coarse,neighbors)->np.ndarray:
    """3xHxW float32. Normalize ONLY current H-alpha ROI and source masks.

    It never accepts annotation masks/GT, IoU score, physical IDs or other
    metadata as forward features. User controls which permitted source ROI is
    passed; provenance is verified by the later dataset-builder stage.
    """
    image,coarse,neighbors=_ensure_source_arrays(image,coarse,neighbors)
    lo,hi=np.percentile(image.astype(np.float32,copy=False),[1,99])
    if hi-lo<=1e-6:
        scaled=np.zeros_like(image,dtype=np.float32)
    else:
        scaled=np.clip((image.astype(np.float32)-lo)/(hi-lo),0,1)
    # np.percentile() produces float64 bounds; clipping then promotes the
    # intensity channel to float64. All model inputs must be torch.float32.
    return np.stack((scaled,coarse.astype(np.float32),
                     neighbors.astype(np.float32)),axis=0).astype(
                         np.float32, copy=False)


class _ConvBlock(nn.Module):
    def __init__(self,cin:int,cout:int):
        super().__init__()
        if cout%8:
            raise ValueError("GroupNorm channels must be divisible by 8")
        self.layers=nn.Sequential(
            nn.Conv2d(cin,cout,3,padding=1,bias=False),
            nn.GroupNorm(8,cout),nn.SiLU(),
            nn.Conv2d(cout,cout,3,padding=1,bias=False),
            nn.GroupNorm(8,cout),nn.SiLU()
        )

    def forward(self,x):
        return self.layers(x)


class NativeROIRefiner(nn.Module):
    """One-channel residual correction with hard forbidden-neighbor pixels.

    Three image-only inference channels: native H-alpha, coarse mask, neighbors.
    Does not alter source instance identity or infer extra objects.
    """
    def __init__(self):
        super().__init__()
        self.e1=_ConvBlock(3,16)
        self.e2=_ConvBlock(16,32)
        self.e3=_ConvBlock(32,64)
        self.bridge=_ConvBlock(64,96)
        self.d3=_ConvBlock(96+64,64)
        self.d2=_ConvBlock(64+32,32)
        self.d1=_ConvBlock(32+16,16)
        self.delta=nn.Conv2d(16,1,1)

    def forward(self,x):
        if (x.ndim!=4 or x.shape[1]!=INPUT_CHANNELS or
                x.shape[-2]%8 or x.shape[-1]%8 or
                min(x.shape[-2:])<16):
            raise ValueError("Expected [B,3,H,W] native patches divisible by 8")
        a=self.e1(x)
        b=self.e2(F.avg_pool2d(a,2))
        c=self.e3(F.avg_pool2d(b,2))
        d=self.bridge(F.avg_pool2d(c,2))
        z=self.d3(torch.cat((F.interpolate(
            d,size=c.shape[-2:],mode="bilinear",align_corners=False),c),dim=1))
        z=self.d2(torch.cat((F.interpolate(
            z,size=b.shape[-2:],mode="bilinear",align_corners=False),b),dim=1))
        z=self.d1(torch.cat((F.interpolate(
            z,size=a.shape[-2:],mode="bilinear",align_corners=False),a),dim=1))
        source_binary=x[:,1:2]
        prior=(source_binary*2.0-1.0)*REFINER_OUTPUT_PRIOR_LOGIT
        logits=prior+self.delta(z)
        forbidden=(x[:,2:3]>.5)
        return torch.where(
            forbidden,torch.full_like(logits,NEIGHBOR_FORBIDDEN_LOGIT),logits
        )


class OOFMaskQualityHead(nn.Module):
    """Predict IoU of an INPUT candidate from source-only ROI channels.

    Its parameters are RANDOM until future genuine group-disjoint OOF
    candidates and TRAIN-only IoU supervision have been provided. Never use
    untrained predictions to promote an A6 mask to a Kaggle submission.
    """
    def __init__(self):
        super().__init__()
        self.blocks=nn.Sequential(
            _ConvBlock(3,16),nn.AvgPool2d(2),
            _ConvBlock(16,32),nn.AvgPool2d(2),
            _ConvBlock(32,64),nn.AdaptiveAvgPool2d(1),
        )
        self.mlp=nn.Sequential(
            nn.Flatten(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1)
        )

    def forward(self,x):
        if x.ndim!=4 or x.shape[1]!=3 or min(x.shape[-2:])<16:
            raise ValueError("Quality needs 3 native-pixel candidate channels")
        return self.mlp(self.blocks(x)).squeeze(1)


def choose_mask_by_oof_quality(image,original,corrected,neighbors,
                               trained_quality_head:OOFMaskQualityHead,
                               *, oof_training_manifest_verified:bool):
    """Inference-only decision with no GT: source data + trained OOF head.

    Fail closed if proof of OOF quality training is absent. One fixed +.03
    quality improvement bar, not picked by Kaggle public leaderboard.
    """
    if not oof_training_manifest_verified:
        raise RuntimeError("Quality selection requires verified OOF TRAIN proposal provenance")
    im,old,near=_ensure_source_arrays(image,original,neighbors)
    _,new,_=_ensure_source_arrays(image,corrected,neighbors)
    if not bool(np.any(new)):
        return old.astype(np.uint8),{"choice":"original_empty_refinement"}
    source_old=torch.from_numpy(source_roi_channels(im,old,near))[None]
    source_new=torch.from_numpy(source_roi_channels(im,new,near))[None]
    params=list(trained_quality_head.parameters())
    device=params[0].device if params else torch.device("cpu")
    trained_quality_head.eval()
    with torch.no_grad():
        qold=float(torch.sigmoid(
            trained_quality_head(source_old.to(device)))[0].item())
        qnew=float(torch.sigmoid(
            trained_quality_head(source_new.to(device)))[0].item())
    if not np.isfinite([qold,qnew]).all():
        raise RuntimeError("Nonfinite OOF quality scores")
    if qnew >= qold + QUALITY_ADVANTAGE_MIN:
        return new.astype(np.uint8),{
            "choice":"refined", "q_original":qold,"q_refined":qnew}
    return old.astype(np.uint8),{
        "choice":"original_quality_margin",
        "q_original":qold,"q_refined":qnew}
