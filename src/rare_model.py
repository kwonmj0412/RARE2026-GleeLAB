#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rare_model.py — RARE26 제출용 모델 + 전처리 (학습과 바이트 단위 일치)
=====================================================================
train_baseline.py 의 Net / fov_bbox / load_pair 정규화를 그대로 이식한다.
전처리 한 줄이라도 학습과 다르면 검증 fpr90 0.03 모델이 리더보드에서 무너진다.

핵심 일치 항목:
  - FOV bbox crop -> 512 resize (INTER_AREA)   [학습 load_pair 와 동일]
  - RGB, /255, ImageNet mean/std               [학습과 동일]
  - top-k pooling (k = topk_frac * h * w, 기본 2%)  mask_aware=False -> 전 위치 대상
  - GC 입력은 이미 RGB uint8 (probe 로 확인). 추가 채널 변환 없음.

체크포인트: train_baseline.py 가 저장한 best.pt (state_dict). Net 구조와 정확히 대응.
5-fold(또는 임의 개수) 가중치를 받아 logit 평균 앙상블.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

cv2.setNumThreads(0)
IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)


# ---------------------------------------------------------------- 전처리 (학습과 동일)
def fov_bbox(img: np.ndarray):
    """train_baseline.fov_bbox 와 동일. img 는 HWC (RGB 또는 BGR 무관 — max 로 판정)."""
    g = img.max(axis=2)
    fg = cv2.morphologyEx((g > 18).astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0, 0, img.shape[1], img.shape[0]
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return (0, 0, img.shape[1], img.shape[0]) if (w < 32 or h < 32) else (x, y, w, h)


def shades_of_gray(img: np.ndarray, p: int = 6, eps: float = 1e-6) -> np.ndarray:
    """train_baseline.shades_of_gray 와 동일. color_const='sog' 인 체크포인트용."""
    f = img.astype(np.float32)
    m = f.max(2) > 18
    if m.sum() < 100:
        return img
    illum = np.array([np.power(np.mean(np.power(f[..., c][m], p)), 1.0 / p) for c in range(3)])
    illum = illum / (np.linalg.norm(illum) / np.sqrt(3.0) + eps)
    return np.clip(f / (illum[None, None, :] + eps), 0, 255).astype(np.uint8)


def white_balance_pct(img: np.ndarray, pct: float = 95.0, dark_thr: int = 20) -> np.ndarray:
    """train_baseline.white_balance_pct 와 동일. FocalScope 화이트밸런스. RGB uint8."""
    out = img.astype(np.float32)
    for c in range(3):
        ch = out[..., c]
        valid = ch[ch > dark_thr]
        if valid.size > 0:
            target = np.percentile(valid, pct)
            if target > 0:
                out[..., c] = ch * (128.0 / target)
    return np.clip(out, 0, 255).astype(np.uint8)


def clahe_lab(img: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """train_baseline.clahe_lab 와 동일. LAB 의 L 에만 CLAHE. RGB uint8."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


def preprocess_frame(rgb: np.ndarray, size: int = 512, fov_crop: bool = True,
                     color_const: str = "none") -> np.ndarray:
    """GC 입력 프레임(HWC, RGB, uint8) -> 정규화된 CHW float32.
    학습 load_pair 와 동일한 순서: FOV crop -> color_const -> resize(AREA) -> /255 -> (x-mean)/std
    (주의: 학습은 FOV crop 후 color_const 적용. WB/CLAHE 는 검은 테두리에 민감하므로 순서 필수 일치.)
    """
    img = rgb
    if fov_crop:
        x, y, w, h = fov_bbox(img)
        img = img[y:y + h, x:x + w]
    if color_const == "sog":
        img = shades_of_gray(img)
    elif color_const == "grayworld":
        img = shades_of_gray(img, p=1)
    elif color_const == "wb":
        img = white_balance_pct(img)
    elif color_const == "clahe":
        img = clahe_lab(img)
    elif color_const == "wb_clahe":
        img = clahe_lab(white_balance_pct(img))
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    x = (img.astype(np.float32) / 255.0 - IMNET_MEAN) / IMNET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


# ---------------------------------------------------------------- 모델 (학습과 동일)
def _strip_prefix(sd: dict) -> dict:
    for p in ("module.", "backbone.", "encoder_q.", "encoder.", "model.", "student.", "teacher."):
        if sd and all(k.startswith(p) for k in sd):
            sd = {k[len(p):]: v for k, v in sd.items()}
    return sd


class LoRALinear(nn.Module):
    """학습 때와 동일 구조 (base 동결 + A/B). best.pt 로딩 호환용."""
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        self.r, self.scale = r, alpha / r
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.scale


def _apply_lora(vit, r=8, alpha=16, targets=("qkv",)):
    for blk in vit.blocks:
        for name in targets:
            mod = getattr(blk.attn, name, None)
            if isinstance(mod, nn.Linear):
                setattr(blk.attn, name, LoRALinear(mod, r, alpha))
    return vit


class Net(nn.Module):
    """train_baseline.Net 와 구조적으로 동일. best.pt state_dict 를 로딩.
    backbone: resnet50(2048) / convnext_base(1024) / dinov3_vitl(1024, LoRA)."""
    def __init__(self, pool: str = "topk", topk_frac: float = 0.02, mask_aware: bool = False,
                 backbone: str = "resnet50", lora_r: int = 8):
        super().__init__()
        self.backbone_name = backbone
        self.vit = None
        if backbone == "dinov3_vitl":
            import timm
            # pretrained=False: best.pt 가 LoRA 포함 전체 가중치를 담고 있음
            self.vit = timm.create_model("vit_large_patch16_dinov3", pretrained=False,
                                         num_classes=0, dynamic_img_size=True)
            self.n_prefix = getattr(self.vit, "num_prefix_tokens", 1)
            self.vit = _apply_lora(self.vit, r=lora_r, alpha=lora_r * 2)
            self.body = None
            feat_ch = self.vit.embed_dim          # 1024
        elif backbone == "convnext_base":
            import timm
            self.body = timm.create_model("convnext_base", pretrained=False,
                                          features_only=True, out_indices=(3,))
            feat_ch = 1024
        elif backbone == "vits_gastronet":
            import timm
            # ViT-Small patch16, embed 384. best.pt 가 전체 가중치 담음 (LoRA 없음).
            self.vit = timm.create_model("vit_small_patch16_224", pretrained=False,
                                         num_classes=0, dynamic_img_size=True)
            self.n_prefix = getattr(self.vit, "num_prefix_tokens", 1)
            self.body = None
            feat_ch = self.vit.embed_dim          # 384
        elif backbone == "maxvit_tiny":
            import timm
            # MaxViT-Tiny (512 전용). best.pt 가 전체 가중치 담음 -> pretrained=False.
            self.body = timm.create_model("maxvit_tiny_tf_512", pretrained=False,
                                          features_only=True, out_indices=(4,))
            feat_ch = self.body.feature_info.channels()[-1]   # 512
        elif backbone == "caformer_s18":
            import timm
            # CAFormer-S18 hybrid conv-transformer. best.pt 가 전체 가중치 담음.
            # num_classes=0, img_size=768. _body 에서 forward_features 사용.
            self.body = timm.create_model("caformer_s18", pretrained=False,
                                          num_classes=0, img_size=768)
            feat_ch = 512
        else:
            from torchvision.models import resnet50
            base = resnet50(weights=None)
            self.body = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool,
                                      base.layer1, base.layer2, base.layer3, base.layer4)
            feat_ch = 2048
        self.head = nn.Sequential(nn.Conv2d(feat_ch, 256, 1), nn.GroupNorm(8, 256),
                                  nn.GELU(), nn.Conv2d(256, 1, 1))
        self.pool, self.topk_frac, self.mask_aware = pool, topk_frac, mask_aware

    def _body(self, x):
        if self.vit is not None:
            tok = self.vit.forward_features(x)       # (B, n_prefix + h*w, C)
            patch = tok[:, self.n_prefix:, :]
            B, N, C = patch.shape
            g = int(round(N ** 0.5))
            return patch.transpose(1, 2).reshape(B, C, g, g).contiguous()
        if self.backbone_name == "caformer_s18":
            return self.body.forward_features(x)     # (B,512,24,24) NCHW
        f = self.body(x)
        if isinstance(f, (list, tuple)):
            f = f[-1]
        return f

    def forward(self, x, mask=None):
        f = self._body(x)
        lm = self.head(f)                      # B,1,h,w
        B, _, h, w = lm.shape
        if mask is None:
            mask = torch.zeros(B, 1, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
        ms = F.interpolate(mask, size=(h, w), mode="area")
        valid = (ms < 0.5) if self.mask_aware else torch.ones_like(ms, dtype=torch.bool)
        allbad = valid.flatten(1).sum(1) == 0
        if allbad.any():
            valid[allbad] = True

        if self.pool == "gap":
            v = (lm * valid).flatten(1).sum(1) / valid.flatten(1).sum(1).clamp(min=1)
        elif self.pool == "lse":
            z = lm.masked_fill(~valid, -1e4).flatten(1)
            nvalid = valid.flatten(1).sum(1).clamp(min=1).to(z.dtype)
            v = torch.logsumexp(z * 4.0, 1) / 4.0 - torch.log(nvalid) / 4.0
        else:  # topk
            z = lm.masked_fill(~valid, -1e4).flatten(1)
            k = max(1, int(round(self.topk_frac * h * w)))
            k = min(k, int(valid.flatten(1).sum(1).min().item()))
            v = z.topk(max(k, 1), dim=1).values.mean(1)
        return v  # logit (B,)


class Ensemble:
    """여러 best.pt 를 로드해 logit 평균. TTA(기하 flip) 옵션."""
    def __init__(self, ckpt_paths: List[Path], pool="topk", topk_frac=0.02,
                 mask_aware=False, device="cuda", tta=True):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tta = tta
        self.models: List[Net] = []
        for p in ckpt_paths:
            # 파일명으로 backbone 감지: cnx/convnext 포함시 ConvNeXt, 아니면 ResNet50
            fname = Path(p).name.lower()
            if "vits" in fname:
                bb = "vits_gastronet"
            elif "dinov3" in fname or "dv3" in fname:
                bb = "dinov3_vitl"
            elif "caformer" in fname or "caf" in fname:
                bb = "caformer_s18"
            elif "maxvit" in fname or "mvit" in fname:
                bb = "maxvit_tiny"
            elif "cnx" in fname or "convnext" in fname:
                bb = "convnext_base"
            else:
                bb = "resnet50"
            m = Net(pool, topk_frac, mask_aware, backbone=bb)
            sd = torch.load(str(p), map_location="cpu", weights_only=False)
            for key in ("state_dict", "model"):
                if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
                    sd = sd[key]
            sd = _strip_prefix(dict(sd))
            missing, unexpected = m.load_state_dict(sd, strict=False)
            n_ok = len(m.state_dict()) - len(missing)
            print(f"[ckpt] {Path(p).name} ({bb}): {n_ok}/{len(m.state_dict())} loaded"
                  f"{' MISSING=' + str(len(missing)) if missing else ''}"
                  f"{' UNEXPECTED=' + str(len(unexpected)) if unexpected else ''}")
            if len(missing) > 5:
                print(f"  [!] 미매칭 {len(missing)}개. 체크포인트-모델 구조 불일치 가능. "
                      f"pool/topk_frac/mask_aware 인자를 학습과 맞추세요.")
            m.to(self.device).eval()
            self.models.append(m)
        if not self.models:
            raise RuntimeError("로드된 체크포인트가 없습니다.")
        print(f"[ensemble] {len(self.models)} models, tta={tta}, device={self.device}")

    @torch.no_grad()
    def logits(self, batch_chw: np.ndarray) -> np.ndarray:
        """batch_chw: (B,3,512,512) float32 정규화 완료. 반환: (B,) 평균 logit."""
        x = torch.from_numpy(batch_chw).to(self.device)
        views = [x]
        if self.tta:
            views += [torch.flip(x, dims=[3]), torch.flip(x, dims=[2]),
                      torch.flip(x, dims=[2, 3])]
        acc = torch.zeros(x.shape[0], device=self.device, dtype=torch.float32)
        n = 0
        use_amp = self.device.type == "cuda"
        for m in self.models:
            for v in views:
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    out = m(v.contiguous(memory_format=torch.channels_last))
                acc += out.float()
                n += 1
        return (acc / n).cpu().numpy()

    @torch.no_grad()
    def logits_multiscale(self, frames_rgb, model_sizes, model_weights,
                          model_ttas=None, fov_crop=True, color_const="none",
                          fusion="logit", model_ccs=None):
        """이중백본용: 원본 프레임을 모델별 해상도로 각각 전처리 후 가중 평균.
        frames_rgb: list of (H,W,3) uint8 RGB (원본).
        model_sizes: 모델별 학습 해상도 리스트 (예: [768,768,...,512,512,...]).
        model_weights: 모델별 가중치 (합=1 아니어도 됨, 정규화함).
        model_ttas: 모델별 TTA 뷰 수 (1 또는 4). None 이면 self.tta 로 일괄.
        model_ccs: 모델별 color_const (예: ['none','none','none','wb_clahe']).
                   None 이면 전역 color_const 를 전 모델에 적용 (하위호환).
                   하이브리드(원본+WB+CLAHE 혼합) 앙상블용.
        비대칭 TTA: RN50(가벼움) TTA4 + DINOv3(무거움) TTA1 -> 시간 절약하며 TTA 이득."""
        from rare_model import preprocess_frame
        B = len(frames_rgb)
        if model_ccs is None:
            model_ccs = [color_const] * len(self.models)
        # (size, cc) 조합별로 전처리 (중복 조합은 1회만)
        uniq_keys = sorted(set(zip(model_sizes, model_ccs)))
        pre = {}
        for sz, cc in uniq_keys:
            arr = np.stack([preprocess_frame(f, sz, fov_crop, cc) for f in frames_rgb])
            pre[(sz, cc)] = torch.from_numpy(arr).to(self.device)
        if model_ttas is None:
            model_ttas = [(4 if self.tta else 1)] * len(self.models)
        acc = torch.zeros(B, device=self.device, dtype=torch.float32)
        wsum = 0.0
        use_amp = self.device.type == "cuda"

        def tta_views(x, n):
            vs = [x]
            if n >= 2:
                vs.append(torch.flip(x, dims=[3]))
            if n >= 4:
                vs += [torch.flip(x, dims=[2]), torch.flip(x, dims=[2, 3])]
            return vs

        for m, sz, w, nt, cc in zip(self.models, model_sizes, model_weights, model_ttas, model_ccs):
            x = pre[(sz, cc)]
            views = tta_views(x, nt)
            # 모델 내부에서 TTA 뷰 평균 -> 그 모델의 대표 logit -> 가중합
            mv = torch.zeros(B, device=self.device, dtype=torch.float32)
            for v in views:
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    out = m(v.contiguous(memory_format=torch.channels_last))
                mv += out.float()
            mv /= len(views)
            # fusion="prob": 각 모델을 확률로 변환 후 가중평균.
            #   mean(sigmoid(x)) != sigmoid(mean(x)) 이므로 순위가 바뀜.
            #   한 모델이 극단적으로 낮은 로짓을 준 양성의 손상을 제한 -> 꼬리 구제.
            acc += (torch.sigmoid(mv) if fusion == "prob" else mv) * w
            wsum += w
        fused = acc / max(wsum, 1e-9)
        if fusion == "prob":
            # 하류에서 sigmoid 를 한 번 더 적용하므로 로짓으로 되돌림 (순위 동일)
            fused = torch.log(fused.clamp(1e-7, 1 - 1e-7) / (1 - fused.clamp(1e-7, 1 - 1e-7)))
        return fused.cpu().numpy()

    @torch.no_grad()
    def dinov3_patch_features(self, frames_rgb, size, fov_crop=True, color_const="none",
                              topk_frac=0.02, max_models=None):
        """배치적응 이상탐지용: DINOv3 모델들의 top-k 패치 평균 특징 (모델간 평균).
        head 이전 _body 표현을 쓴다 (분류 로짓이 아니라 표현 자체).
        max_models: 특징 추출에 쓸 DINOv3 모델 수 제한 (시간 절약).
                    이상탐지는 소수 모델로도 AUROC 0.99 -> 1~2개면 충분.
        반환: (N, C) 이미지별 특징."""
        from rare_model import preprocess_frame
        dv3_models = [m for m in self.models
                      if getattr(m, "backbone_name", "") == "dinov3_vitl"]
        if not dv3_models:
            return None
        if max_models is not None:
            dv3_models = dv3_models[:max_models]   # 앞쪽 소수만 (시간 절약)
        arr = np.stack([preprocess_frame(f, size, fov_crop, color_const) for f in frames_rgb])
        x = torch.from_numpy(arr).to(self.device)
        feats = []
        use_amp = self.device.type == "cuda"
        for m in dv3_models:
            fb = []
            for i in range(0, len(x), 8):
                xb = x[i:i+8].contiguous(memory_format=torch.channels_last)
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    f = m._body(xb)              # (b, C, h, w)
                    lm = m.head(f)               # (b, 1, h, w)
                B, C, h, w = f.shape
                k = max(1, int(round(topk_frac * h * w)))
                fl = f.flatten(2)                # (b, C, h*w)
                ll = lm.flatten(2)[:, 0, :]      # (b, h*w)
                idx = ll.topk(k, dim=1).indices  # (b, k)
                # 각 이미지의 top-k 위치 특징 평균
                gathered = torch.gather(fl, 2, idx.unsqueeze(1).expand(-1, C, -1))  # (b,C,k)
                fb.append(gathered.mean(2).float().cpu())
            feats.append(torch.cat(fb, 0).numpy())
        return np.mean(feats, axis=0)            # 모델간 평균 (N, C)


def batch_adaptive_gate(logits, feats, q=0.6, gate_lo=0.4, penalty=2.0, shrink=0.1,
                        return_debug=False):
    """배치적응 비대칭 게이트 (Step 29).
    테스트 스택 전체를 하나의 배치로 보고:
      1. 분류 로짓 하위 q 를 '이 배치의 정상'으로 -> 정상 다양체(마할라노비스) 구성
      2. 각 이미지의 이상 백분위 계산
      3. 이상 백분위 < gate_lo (정상다운) 이미지의 로짓만 하향 (비대칭)
    병변/애매 이미지(백분위 높음)는 손대지 않음 -> 민감도 보존.
    로컬 검증: 게이트 영역 병변 0%, 양성 로짓 변경 0, FPR90 무해."""
    import numpy as np
    if feats is None or len(logits) < 30:
        return (logits, None) if return_debug else logits
    logits = np.asarray(logits, np.float64)
    N, C = feats.shape
    # 1. 정상 뱅크 (분류 하위 q)
    thr = np.quantile(logits, q)
    bank = feats[logits <= thr]
    if len(bank) < 20:
        return logits
    # 2. 마할라노비스 이상점수 (뱅크 표본 < 차원이면 PCA 로 안정화)
    mu = bank.mean(0)
    Xc = feats - mu
    Bc = bank - mu
    max_dim = max(8, min(C, len(bank) // 3))   # 뱅크 표본의 1/3 이하 차원
    if max_dim < C:
        # 뱅크 공분산의 주성분으로 투영 (표본 부족 시 공분산 안정화)
        U, S, Vt = np.linalg.svd(Bc, full_matrices=False)
        proj = Vt[:max_dim].T                  # (C, max_dim)
        Bc = Bc @ proj
        Xc = Xc @ proj
    cov = np.cov(Bc.T)
    d = cov.shape[0]
    cov = (1 - shrink) * cov + shrink * np.eye(d) * np.trace(cov) / d
    inv = np.linalg.pinv(cov)
    anom = np.sqrt(np.einsum("ij,jk,ik->i", Xc, inv, Xc))
    # 3. 이상 백분위 -> 비대칭 게이트
    apct = anom.argsort().argsort() / max(len(anom) - 1, 1)
    out = logits.copy()
    m = apct < gate_lo
    out[m] = logits[m] - penalty * (gate_lo - apct[m]) / gate_lo
    print(f"[gate] 뱅크 {len(bank)}장(하위 {q:.0%}), 게이트 적용 {int(m.sum())}장 "
          f"(백분위<{gate_lo}), 최대 하향 {penalty}")
    if return_debug:
        return out.astype(np.float32), {"apct": apct, "anom": anom, "gated_mask": m,
                                        "bank_size": len(bank)}
    return out.astype(np.float32)
