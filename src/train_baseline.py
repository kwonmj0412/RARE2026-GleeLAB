#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_baseline.py — RARE26 베이스라인 + shortcut 절제 실험 (Step 4)
===================================================================

검증할 가설
----------
Step 2 에서 측정된 사실:
    AUC(inpaint footprint -> center) = 0.988
    AUC(center            -> label ) = 0.685   (c1 유병률 2.68% vs c2 11.89%)
따라서 순진하게 학습하면 모델은 "매끈한 패치 있음 -> 양성 확률 12%" 라는 규칙으로
AUROC 0.685 를 공짜로 얻는다. 테스트셋에는 인페인팅이 없으므로 이 신호는 증발하고,
그 사전확률에 기대어 학습되지 않은 center_2 양성(전체의 61%)이 tail 로 떨어진다.

세 가지 처방:
    P0-a  center-conditional balanced sampler  -> shortcut 의 **보상**을 0 으로
    P0-b  on-the-fly transplant augmentation   -> shortcut 의 **단서**를 무작위화
    P0-c  mask-aware top-k pooling             -> 환각 픽셀이 점수에 기여 불가

측정 방법
--------
  LOCO:  train center_1 -> eval center_2   /   train center_2 -> eval center_1
  지표:  pAUC(TPR>=0.80)  [모델 선택]      FPR@TPR90  [최종 확인]
  진단:  --footprint-test  같은 이미지에 랜덤 transplant 를 씌우고 점수 변화 측정
                            (불변이어야 정상. |Δlogit| 이 크면 footprint 의존)
         AUC(score -> center) on NDBE only   (0.5 여야 정상. shortcut 직접 측정)

빠른 시작 (절제 4종 × LOCO 2방향 = 8런, Blackwell 기준 ~2시간)
-------------------------------------------------------------
    D=<DATA_ROOT>

    for CFG in naive balance balance_transplant full; do
      for C in center_1 center_2; do
        python train_baseline.py --preset $CFG \
          --data-root "$D/clean" --folds ./folds_step3/folds.csv \
          --mask-bank ./audit_step2/mask_bank \
          --weights "$D/Gastronet-5M-pretrained-models/RN50_Billion-Scale-SWSL%2BGastroNet-5M_DINOv1.pth" \
          --split loco --loco-train $C --epochs 15 --size 512 --bs 16 \
          --footprint-test --out-dir ./runs
      done
    done

    # 요약
    python train_baseline.py --summarize ./runs

그 다음: 최선 설정으로 5-fold OOF
    python train_baseline.py --preset full --split kfold --fold all ... --out-dir ./runs_oof
    python rare_metrics.py --csv ./runs_oof/full_kfold_oof/preds.csv --group-col center

주의: 데이터가 USB 외장 드라이브에 있으면 I/O 가 학습을 지배합니다.
      --cache (기본 on) 로 RAM 에 한 번만 올립니다. 512px 기준 약 3.2GB.

의존성: torch, torchvision, opencv-python, numpy, pandas, scikit-learn, tqdm
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, ConcatDataset
from tqdm import tqdm

cv2.setNumThreads(0)
IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)


# ======================================================================================
# 지표 (rare_metrics.py 와 동일 정의)
# ======================================================================================
def fpr_at_tpr(y, s, target=0.90) -> float:
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
    return float(np.interp(target, tpr, fpr))


def pauc_high_sens(y, s, min_tpr=0.80, n=1001) -> float:
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
    g = np.linspace(min_tpr, 1.0, n)
    return float(np.mean(1.0 - np.interp(g, tpr, fpr)))


def ppv_from_fpr(fpr, tpr=0.90, ratio=100.0) -> float:
    return float(tpr / (tpr + ratio * fpr)) if (tpr + ratio * fpr) > 0 else float("nan")


def boot_ci(y, s, n=400, seed=0):
    """양성·음성 모두 복원추출한 내부 CI.
    val 양성이 ~100장이면 fpr90 의 CI 는 ±0.05 수준으로 넓다.
    arm 간 차이가 이 폭보다 작으면 '차이 없음' 이다."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    f, p = [], []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        yy, ss = y[idx], s[idx]
        if yy.sum() in (0, len(yy)):
            continue
        f.append(fpr_at_tpr(yy, ss)); p.append(pauc_high_sens(yy, ss))
    f, p = np.asarray(f), np.asarray(p)
    return dict(fpr90_lo=float(np.percentile(f, 2.5)), fpr90_hi=float(np.percentile(f, 97.5)),
                pauc80_lo=float(np.percentile(p, 2.5)), pauc80_hi=float(np.percentile(p, 97.5)))


# ======================================================================================
# 전처리 유틸
# ======================================================================================
def fov_bbox(img: np.ndarray) -> Tuple[int, int, int, int]:
    g = img.max(axis=2)
    fg = cv2.morphologyEx((g > 18).astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0, 0, img.shape[1], img.shape[0]
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return (0, 0, img.shape[1], img.shape[0]) if (w < 32 or h < 32) else (x, y, w, h)


def shades_of_gray(img: np.ndarray, p: int = 6, eps: float = 1e-6) -> np.ndarray:
    """Shades-of-Gray 색항등성 (Finlayson & Trezzi). 전역 조명색만 제거하고
    국소 색대비(발적/혈관 패턴 = BING 진단 단서)는 보존한다.

    LAB 전체 표준화와 달리 병변의 국소 붉은기를 지우지 않는다.
    이미지 1장만 보고 계산되므로 추론 컨테이너에서 그대로 사용 가능.
    """
    f = img.astype(np.float32)
    m = f.max(2) > 18                      # FOV 안쪽만 (검은 코너 제외)
    if m.sum() < 100:
        return img
    illum = np.array([np.power(np.mean(np.power(f[..., c][m], p)), 1.0 / p) for c in range(3)])
    illum = illum / (np.linalg.norm(illum) / np.sqrt(3.0) + eps)
    out = f / (illum[None, None, :] + eps)
    return np.clip(out, 0, 255).astype(np.uint8)


def white_balance_pct(img: np.ndarray, pct: float = 95.0, dark_thr: int = 20) -> np.ndarray:
    """FocalScope 화이트밸런스: 채널별 95퍼센타일을 흰점으로 보고 중간회색(128)에 맞춤.
    스코프/센터별 색 편향을 제거해 미지 센터 도메인 시프트를 줄인다.
    검은 FOV 코너(<20)는 흰점 계산에서 제외. 이미지 1장으로 계산 → 추론 안전.
    입력/출력 모두 RGB uint8."""
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
    """FocalScope CLAHE: LAB 공간의 L(명도)에만 적응적 대비 강화.
    조명 차이를 정규화하고 조직 대비를 높인다. a/b(색)는 보존.
    입력/출력 모두 RGB uint8."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


def load_pair(img_path: Path, mask_path: Optional[Path], size: int, fov_crop: bool,
              color_const: str = "none"):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(img_path)
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if (mask_path and mask_path.exists()) else None
    if m is None or m.shape[:2] != img.shape[:2]:
        m = np.zeros(img.shape[:2], np.uint8)
    if fov_crop:
        x, y, w, h = fov_bbox(img)
        img, m = img[y:y + h, x:x + w], m[y:y + h, x:x + w]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    m = (cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
    return img, m


# ======================================================================================
# Dataset
# ======================================================================================
def strong_color_view(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """색불변 consistency 용 강한 색·조명 변형. 형태는 보존, 색온도/밝기/화이트밸런스만 크게.
    미지 센터의 스코프·프로세서 색차를 흉내낸다."""
    f = img.astype(np.float32)
    # 채널별 게인 (화이트밸런스 시프트 = 센터별 색온도)
    for c in range(3):
        f[..., c] *= rng.uniform(0.7, 1.3)
    f *= rng.uniform(0.75, 1.25)                                  # brightness
    f = (f - f.mean()) * rng.uniform(0.75, 1.25) + f.mean()       # contrast
    f = np.clip(f, 0, 255).astype(np.uint8)
    # 감마
    g = rng.uniform(0.7, 1.4)
    lut = np.clip(((np.arange(256)/255.0) ** g) * 255, 0, 255).astype(np.uint8)
    f = lut[f]
    # hue/sat
    hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + rng.randint(-12, 12)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.7, 1.3), 0, 255)
    f = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    # 확률적 SoG (일부 뷰만 색항등성 적용 -> 더 다양한 색조건)
    if rng.random() < 0.5:
        f = shades_of_gray(f)
    return f


class ConsistencyDS(Dataset):
    """unlabeled 식도 프레임(GastroNet). 각 이미지에서 '같은 형태·다른 색' 두 뷰 생성.
    색불변 학습: 두 뷰가 같은 top-k logit 을 갖도록 -> 미지 센터 색차에 강건.
    """
    def __init__(self, files: List[str], size: int, fov_crop: bool = True):
        self.files = files; self.size = size; self.fov_crop = fov_crop

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = cv2.imread(self.files[i], cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((self.size, self.size, 3), np.uint8)
        else:
            if self.fov_crop:
                x, y, w, h = fov_bbox(img)
                img = img[y:y+h, x:x+w]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        # 공통 기하변형 (두 뷰 동일 -> 형태 정렬 유지, 색만 다르게)
        rng = random.Random(i * 7919 + random.randint(0, 1 << 20))
        if rng.random() < 0.5:
            img = img[:, ::-1]
        k = rng.randint(0, 3)
        if k:
            img = np.rot90(img, k)
        img = np.ascontiguousarray(img)
        # 두 색 뷰
        v1 = strong_color_view(img, rng)
        v2 = strong_color_view(img, rng)
        def norm(v):
            x = (v.astype(np.float32)/255.0 - IMNET_MEAN)/IMNET_STD
            return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        z = torch.zeros(1, self.size, self.size)   # 더미 마스크
        return norm(v1), norm(v2), z



def fourier_amp_perturb(img: np.ndarray, L: float = 0.01, eta: float = 0.5,
                        rng: Optional[random.Random] = None) -> np.ndarray:
    """Fourier 진폭 증강 (도메인 랜덤화). 위상(구조/병변)은 보존하고
    저주파 진폭(색/조명/스캐너 스타일)에만 랜덤 스케일 섭동.
    CLAHE 는 고정 색정규화지만, 이건 학습 중 무한한 색/스타일 변형을 생성 =
    미지 센터(12개 BONSAI) 도메인 갭을 학습 시 시뮬레이션. (Xu CVPR2021, FDA CVPR2020)
    L: 섭동할 저주파 대역 비율, eta: 섭동 강도 (1±eta)."""
    if rng is None:
        rng = random.Random()
    f = img.astype(np.float32)
    F = np.fft.fft2(f, axes=(0, 1))
    amp, pha = np.abs(F), np.angle(F)
    h, w = img.shape[:2]
    amp_s = np.fft.fftshift(amp, axes=(0, 1))
    bh, bw = max(1, int(h * L)), max(1, int(w * L))
    ch, cw = h // 2, w // 2
    reg = amp_s[ch - bh:ch + bh + 1, cw - bw:cw + bw + 1]
    # 채널 공통 랜덤 스케일 (색 시프트 유발), 픽셀별 미세 변동
    g = np.random.default_rng(rng.randint(0, 2**31 - 1))
    scale = g.uniform(1 - eta, 1 + eta, reg.shape).astype(np.float32)
    amp_s[ch - bh:ch + bh + 1, cw - bw:cw + bw + 1] = reg * scale
    amp2 = np.fft.ifftshift(amp_s, axes=(0, 1))
    out = np.fft.ifft2(amp2 * np.exp(1j * pha), axes=(0, 1))
    return np.clip(np.real(out), 0, 255).astype(np.uint8)


def strong_color_aug(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """consistency DAPT 용 강한 색 변형. 미지 센터의 색온도/화이트밸런스 변이를 흉내.
    _color 보다 훨씬 세게 흔들어, 모델이 '색 불변'을 배우게 한다."""
    f = img.astype(np.float32)
    f *= rng.uniform(0.6, 1.4)                                    # brightness 크게
    f = (f - f.mean()) * rng.uniform(0.6, 1.4) + f.mean()         # contrast
    if rng.random() < 0.5:                                        # 랜덤 SoG (색항등성)
        f = shades_of_gray(np.clip(f, 0, 255).astype(np.uint8)).astype(np.float32)
    hsv = cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + rng.randint(-25, 25)) % 180      # hue 크게
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.5, 1.5), 0, 255)   # saturation
    hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.7, 1.3), 0, 255)   # value
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    # 채널별 곱 (화이트밸런스 시프트)
    out = np.clip(out.astype(np.float32) *
                  np.array([rng.uniform(0.8, 1.2) for _ in range(3)], np.float32), 0, 255)
    return out.astype(np.uint8)


class RareDS(Dataset):
    def __init__(self, df: pd.DataFrame, root: Path, bank: Optional[Path], size: int,
                 train: bool, cfg: "Cfg", stencils: Optional[np.ndarray], cache: bool):
        self.df = df.reset_index(drop=True)
        self.root, self.bank, self.size = root, bank, size
        self.train, self.cfg, self.stencils = train, cfg, stencils
        # NOTE: 캐시는 반드시 '연속된 단일 numpy 배열'이어야 한다.
        #       파이썬 리스트로 두면 num_workers 개의 fork 프로세스가 refcount 를 건드리며
        #       copy-on-write 페이지를 전부 복제해 RAM 이 워커 수만큼 곱해진다.
        self.cimg: Optional[np.ndarray] = None
        self.cmsk: Optional[np.ndarray] = None
        if cache:
            n = len(self.df)
            self.cimg = np.zeros((n, size, size, 3), np.uint8)
            self.cmsk = np.zeros((n, size, size), np.uint8)
            for j, r in enumerate(tqdm(self.df.itertuples(), total=n, desc="cache", leave=False)):
                self.cimg[j], self.cmsk[j] = self._raw(r.rel)

    def _mask_path(self, rel: str) -> Optional[Path]:
        return (self.bank / rel) if self.bank else None

    def _raw(self, rel: str):
        return load_pair(self.root / rel, self._mask_path(rel), self.size,
                         self.cfg.fov_crop, self.cfg.color_const)

    def _get(self, i: int):
        if self.cimg is not None:
            return self.cimg[i].copy(), self.cmsk[i].copy()
        return self._raw(self.df.iloc[i].rel)

    def __len__(self):
        return len(self.df)

    # ---------------- 증강 ----------------
    def _geom(self, img, m):
        if random.random() < 0.5:
            img, m = img[:, ::-1], m[:, ::-1]
        if random.random() < 0.5:
            img, m = img[::-1], m[::-1]
        k = random.randint(0, 3)
        if k:
            img, m = np.rot90(img, k), np.rot90(m, k)
        img, m = np.ascontiguousarray(img), np.ascontiguousarray(m)
        if random.random() < 0.8:  # random resized crop
            s = random.uniform(0.72, 1.0)
            ar = random.uniform(0.9, 1.11)
            hh, ww = int(self.size * s / math.sqrt(ar)), int(self.size * s * math.sqrt(ar))
            hh, ww = min(hh, self.size), min(ww, self.size)
            y0 = random.randint(0, self.size - hh); x0 = random.randint(0, self.size - ww)
            img = cv2.resize(img[y0:y0 + hh, x0:x0 + ww], (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            m = cv2.resize(m[y0:y0 + hh, x0:x0 + ww], (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        return img, m

    def _color(self, img):
        f = img.astype(np.float32)
        f *= random.uniform(0.85, 1.15)                       # brightness
        f = (f - f.mean()) * random.uniform(0.85, 1.15) + f.mean()   # contrast
        hsv = cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + random.randint(-6, 6)) % 180    # hue
        hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.85, 1.15), 0, 255)
        return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)

    def _sharpness(self, img):
        """구조강조(structure enhancement) 증강.
        내시경 장비(Olympus/Fujifilm/Pentax)의 강조모드 설정 차이는 엣지 샤프닝 강도를
        바꾸며, 같은 병변이 설정만으로 검출<->미검출로 뒤집힌다는 보고가 있다
        (BONSAI, Endoscopy 2025). LAB 전역통계 기반 색 증강으로는 이 축이 커버되지 않는다.

        블러(약한 강조)와 언샤프 마스킹(강한 강조)을 양방향으로 적용해 불변성을 학습.
        색조는 건드리지 않으므로 병변의 미묘한 색 신호는 보존된다.
        """
        r = random.random()
        if r < 0.45:                                   # 블러 (강조 약한 장비 모사)
            sig = random.uniform(0.4, 1.3)
            return cv2.GaussianBlur(img, (0, 0), sig)
        if r < 0.90:                                   # 언샤프 마스킹 (강조 강한 장비)
            sig = random.uniform(0.6, 1.6)
            amt = random.uniform(0.3, 1.1)
            blur = cv2.GaussianBlur(img, (0, 0), sig)
            sharp = img.astype(np.float32) * (1 + amt) - blur.astype(np.float32) * amt
            return np.clip(sharp, 0, 255).astype(np.uint8)
        return img                                     # 10% 는 원본 유지

    def _transplant(self, img, m, rng: Optional[random.Random] = None):
        """스텐실을 랜덤 위치/방향으로 이식하고, 인페인팅과 유사한 '매끈한' 채움을 적용."""
        R = rng or random
        st = self.stencils[R.randrange(len(self.stencils))].copy()
        if R.random() < 0.5:
            st = st[:, ::-1]
        if R.random() < 0.5:
            st = st[::-1]
        st = np.rot90(st, R.randint(0, 3))
        st = np.roll(st, (R.randint(-64, 64), R.randint(-64, 64)), axis=(0, 1))
        st = np.ascontiguousarray(st).astype(np.uint8)
        fill = cv2.blur(img, (31, 31))
        img = np.where(st[..., None] > 0, fill, img)
        return img, np.clip(m + st, 0, 1)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img, m = self._get(i)

        if self.train:
            img, m = self._geom(img, m)
            if self.stencils is not None and random.random() < self.cfg.transplant_p:
                img, m = self._transplant(img, m)
            img = self._color(img)
            if getattr(self.cfg, "fourier_p", 0.0) > 0 and random.random() < self.cfg.fourier_p:
                img = fourier_amp_perturb(img, L=self.cfg.fourier_L,
                                          eta=self.cfg.fourier_eta,
                                          rng=random.Random(random.randint(0, 2**31 - 1)))
            if self.cfg.sharpness_p > 0 and random.random() < self.cfg.sharpness_p:
                img = self._sharpness(img)

        x = (img.astype(np.float32) / 255.0 - IMNET_MEAN) / IMNET_STD
        return (torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))),
                torch.from_numpy(m[None].astype(np.float32)),
                torch.tensor(float(r.label)), i)

    def eval_variant(self, i, seed: int):
        """footprint-test 용: 동일 이미지에 결정론적 transplant 를 씌운 버전."""
        img, m = self._get(i)
        img, m = self._transplant(img, m, random.Random(seed + i))
        x = (img.astype(np.float32) / 255.0 - IMNET_MEAN) / IMNET_STD
        return (torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))),
                torch.from_numpy(m[None].astype(np.float32)))


# ======================================================================================
# 모델
# ======================================================================================
def strip_prefix(sd: dict) -> dict:
    for p in ("module.", "backbone.", "encoder_q.", "encoder.", "model.", "student.", "teacher."):
        if sd and all(k.startswith(p) for k in sd):
            sd = {k[len(p):]: v for k, v in sd.items()}
    return sd


class LoRALinear(nn.Module):
    """nn.Linear 에 저랭크 어댑터. base 는 동결, A/B 만 학습.
    B=0 초기화 -> 학습 시작시 출력이 base 와 동일 (안전).
    저데이터(양성 158)에서 full fine-tune 과적합을 막는 표준 기법.
    MIDOG 2025 우승: DINOv3 + LoRA (~1.3M 만 학습)."""
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r, self.scale = r, alpha / r
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)   # B 는 0 유지

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.scale


def apply_lora(vit, r=8, alpha=16, targets=("qkv",)):
    """ViT blocks 의 지정 Linear 를 LoRALinear 로 교체. 나머지 전부 동결."""
    for p in vit.parameters():
        p.requires_grad = False
    n_lora = 0
    for blk in vit.blocks:
        for name in targets:
            mod = getattr(blk.attn, name, None)
            if isinstance(mod, nn.Linear):
                setattr(blk.attn, name, LoRALinear(mod, r, alpha))
                n_lora += 1
    trainable = sum(p.numel() for p in vit.parameters() if p.requires_grad)
    print(f"[lora] r={r} alpha={alpha} targets={targets}: {n_lora}개 층 교체, "
          f"학습 파라미터 {trainable/1e6:.2f}M / 전체 {sum(p.numel() for p in vit.parameters())/1e6:.1f}M")
    return vit


class Net(nn.Module):
    def __init__(self, weights: Optional[str], pool: str, topk_frac: float, mask_aware: bool,
                 backbone: str = "resnet50", lora_r: int = 8):
        super().__init__()
        self.backbone_name = backbone
        if backbone == "resnet50":
            from torchvision.models import resnet50, ResNet50_Weights
            if weights and weights.lower() != "none":
                base = resnet50(weights=None)
                sd = torch.load(weights, map_location="cpu", weights_only=False)
                for k in ("state_dict", "model", "teacher", "student", "net"):
                    if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
                        sd = sd[k]
                sd = strip_prefix(dict(sd))
                tgt = base.state_dict()
                keep = {k: v for k, v in sd.items() if k in tgt and tgt[k].shape == v.shape}
                base.load_state_dict(keep, strict=False)
                print(f"[backbone] resnet50 {Path(weights).name}: {len(keep)}/{len(tgt)} loaded")
            else:
                base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
                print("[backbone] ImageNet ResNet50")
            self.body = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool,
                                      base.layer1, base.layer2, base.layer3, base.layer4)
            feat_ch = 2048
        elif backbone == "convnext_base":
            import timm
            is_pth = bool(weights and weights.lower() != "none" and weights.endswith(".pth"))
            # features_only 모델 생성 (공간맵용). 사전학습은 아래서 덮어씀.
            self.body = timm.create_model("convnext_base", pretrained=not is_pth,
                                          features_only=True, out_indices=(3,))
            if is_pth:
                # 우리 SSL 사전학습 가중치 (.pth) 로드. num_classes=0 형식 -> features_only 매칭.
                sd = torch.load(weights, map_location="cpu", weights_only=False)
                for k in ("state_dict", "model"):
                    if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
                        sd = sd[k]
                tgt = self.body.state_dict()
                keep = {k: v for k, v in sd.items() if k in tgt and tgt[k].shape == v.shape}
                self.body.load_state_dict(keep, strict=False)
                print(f"[backbone] ConvNeXt-B SSL {Path(weights).name}: {len(keep)}/{len(tgt)} loaded")
            else:
                # timm 태그 (예: fb_in22k)
                tag = weights if (weights and weights.lower() not in ("none",)) else "fb_in22k"
                self.body = timm.create_model(f"convnext_base.{tag}", pretrained=True,
                                              features_only=True, out_indices=(3,))
                print(f"[backbone] ConvNeXt-B ({tag}) features_only")
            feat_ch = 1024
        elif backbone == "maxvit_tiny":
            import timm
            # RARE25 UT팀 백본. hybrid conv-transformer, 저데이터 병변에 강함.
            # ViT 계열(DINOv3)과 아키텍처가 근본적으로 달라 앙상블 상보성 큼.
            is_pt = bool(weights and weights.lower() != "none" and weights.endswith(".pt"))
            self.body = timm.create_model("maxvit_tiny_tf_512", pretrained=not is_pt,
                                          features_only=True, out_indices=(4,))
            if is_pt:
                # GastroNet DINO 도메인 적응 가중치 로드
                sd = torch.load(weights, map_location="cpu", weights_only=False)
                for k in ("teacher", "student", "state_dict", "model", "backbone"):
                    if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
                        sd = sd[k]
                sd = strip_prefix(dict(sd))
                tgt = self.body.state_dict()
                keep = {k: v for k, v in sd.items() if k in tgt and tgt[k].shape == v.shape}
                self.body.load_state_dict(keep, strict=False)
                print(f"[backbone] MaxViT-Tiny + GastroNet {Path(weights).name}: "
                      f"{len(keep)}/{len(tgt)} loaded")
            else:
                print(f"[backbone] MaxViT-Tiny (tf_512, ImageNet)")
            feat_ch = self.body.feature_info.channels()[-1]  # 512
        elif backbone == "caformer_s18":
            import timm
            # Insights-CADe-BE (Kusters CMPB2025, 주최 그룹) 최강 단일 백본.
            # 4-stage hybrid: stage0-1 = SepConv(국소/CNN), stage2-3 = Attention(전역/Transformer).
            # 단일 모델 내 CNN+Transformer 융합 -> RN50/ViT 앙상블과 다른 표현 = 상보성.
            is_pt = bool(weights and weights.lower() != "none" and weights.endswith((".pt", ".pth")))
            # in22k_ft_in1k timm 태그를 weights 로 명시했을 때만 pretrained 다운로드.
            is_timm_tag = bool(weights and weights.lower() not in ("none",)
                               and not weights.endswith((".pt", ".pth")))
            if is_timm_tag:
                # 학습 시작용: timm ImageNet 사전학습 다운로드
                self.body = timm.create_model(f"caformer_s18.{weights}", pretrained=True,
                                              num_classes=0, img_size=768)
                print(f"[backbone] CAFormer-S18 ({weights}) hybrid conv-transformer")
            else:
                # 기본(weights=none): 뼈대만. best.pt(전체 가중치)가 이후 덮어씀.
                # 하네스/추론/기본학습 모두 여기 -> 인터넷 다운로드 불필요.
                self.body = timm.create_model("caformer_s18", pretrained=False,
                                              num_classes=0, img_size=768)
                if is_pt:
                    sd = torch.load(weights, map_location="cpu", weights_only=False)
                    for k in ("teacher", "student", "state_dict", "model", "backbone"):
                        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
                            sd = sd[k]
                    sd = strip_prefix(dict(sd))
                    tgt = self.body.state_dict()
                    keep = {k: v for k, v in sd.items() if k in tgt and tgt[k].shape == v.shape}
                    self.body.load_state_dict(keep, strict=False)
                    print(f"[backbone] CAFormer-S18 + SSL {Path(weights).name}: "
                          f"{len(keep)}/{len(tgt)} loaded")
                else:
                    print(f"[backbone] CAFormer-S18 (skeleton, best.pt 로 로드)")
            feat_ch = 512
        elif backbone == "dinov3_vitl":
            import timm
            # RARE25 우승(IMSY) 핵심 구성요소. DINOv3 ViT-L/16, LVD-1689M 사전학습.
            # 512px -> 32x32 격자 (RN50 의 16x16 보다 4배 조밀 = 격자 이득 공짜).
            self.vit = timm.create_model("vit_large_patch16_dinov3", pretrained=True,
                                         num_classes=0, dynamic_img_size=True)
            self.n_prefix = getattr(self.vit, "num_prefix_tokens", 1)  # CLS 1 + register 4 = 5
            # GastroNet DINO continued-pretraining 백본으로 교체 (있으면). LoRA 전에 적용.
            if weights and weights.lower() != "none" and weights.endswith(".pt"):
                sd = torch.load(weights, map_location="cpu", weights_only=False)
                for k in ("teacher", "student", "state_dict", "model", "backbone"):
                    if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
                        sd = sd[k]
                sd = strip_prefix(dict(sd))
                tgt = self.vit.state_dict()
                keep = {k: v for k, v in sd.items() if k in tgt and tgt[k].shape == v.shape}
                miss = self.vit.load_state_dict(keep, strict=False)
                print(f"[backbone] DINOv3 + GastroNet-DINO {Path(weights).name}: "
                      f"{len(keep)}/{len(tgt)} loaded")
            else:
                print(f"[backbone] DINOv3 ViT-L/16 (lvd1689m), prefix_tokens={self.n_prefix}")
            self.vit = apply_lora(self.vit, r=lora_r, alpha=lora_r * 2)
            self.body = None
            feat_ch = self.vit.embed_dim          # 1024
        elif backbone == "vits_gastronet":
            import timm
            # ViT-Small patch16. dynamic_img_size 로 512 입력시 pos_embed 자동 보간.
            self.vit = timm.create_model("vit_small_patch16_224", pretrained=False,
                                         num_classes=0, dynamic_img_size=True)
            self.n_prefix = getattr(self.vit, "num_prefix_tokens", 1)  # CLS 제외용
            if weights and weights.lower() != "none" and weights.endswith(".pth"):
                sd = torch.load(weights, map_location="cpu", weights_only=False)
                for k in ("teacher", "student", "state_dict", "model"):
                    if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
                        sd = sd[k]
                sd = strip_prefix(dict(sd))
                tgt = self.vit.state_dict()
                keep = {k: v for k, v in sd.items() if k in tgt and tgt[k].shape == v.shape}
                self.vit.load_state_dict(keep, strict=False)
                print(f"[backbone] ViT-S GastroNet {Path(weights).name}: {len(keep)}/{len(tgt)} loaded")
            else:
                print("[backbone] ViT-S (random init)")
            self.body = None       # _body 에서 vit 경로 사용
            feat_ch = 384
        else:
            raise ValueError(f"unknown backbone {backbone}")
        self.feat_ch = feat_ch
        self.head = nn.Sequential(nn.Conv2d(feat_ch, 256, 1), nn.GroupNorm(8, 256),
                                  nn.GELU(), nn.Conv2d(256, 1, 1))
        self.pool, self.topk_frac, self.mask_aware = pool, topk_frac, mask_aware

    def _body(self, x):
        """백본 공간 feature map (B,C,h,w) 반환.
        - resnet/convnext: 자연스러운 공간맵
        - vit: patch token (CLS 제외)을 (h,w) 격자로 reshape"""
        if self.backbone_name in ("vits_gastronet", "dinov3_vitl"):
            tok = self.vit.forward_features(x)          # (B, n_prefix + h*w, C)
            patch = tok[:, self.n_prefix:, :]           # CLS/register 제외 -> (B, h*w, C)
            B, N, C = patch.shape
            g = int(round(N ** 0.5))                    # 정사각 격자 가정
            f = patch.transpose(1, 2).reshape(B, C, g, g)  # (B, 384, g, g)
            return f.contiguous()
        if self.backbone_name == "caformer_s18":
            # num_classes=0 이면 forward()=pooled 벡터. 공간맵은 forward_features.
            # CAFormer forward_features -> (B, 512, 24, 24) NCHW. top-k pooling 바로 호환.
            return self.body.forward_features(x)
        f = self.body(x)
        if isinstance(f, (list, tuple)):
            f = f[-1]
        return f

    def feat(self, x):
        """consistency DAPT 용: global average pooled backbone feature (B, feat_ch)."""
        f = self._body(x)
        return F.adaptive_avg_pool2d(f, 1).flatten(1)

    def forward(self, x, mask, return_map=False):
        f = self._body(x)                      # B,C,h,w
        lm = self.head(f)                      # B,1,h,w
        B, _, h, w = lm.shape
        ms = F.interpolate(mask, size=(h, w), mode="area")
        valid = (ms < 0.5) if self.mask_aware else torch.ones_like(ms, dtype=torch.bool)
        # 전부 마스킹된 극단 케이스 방어
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
        if return_map:
            # seg 보조용: 분류 logit + 공간 logit map (EVC seg loss 계산에 사용)
            return v, lm
        return v


# ======================================================================================
# 손실
# ======================================================================================
class EMA:
    """가중치 지수이동평균. epoch 간 fpr90 이 0.03<->0.11 로 튀는 것을 억제한다.
    val 양성이 ~30장이라 단일 epoch 선택은 노이즈다 -> 평활화된 가중치로 평가."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                s.copy_(v)

    def copy_to(self, model: nn.Module):
        sd = model.state_dict()
        model.load_state_dict({k: self.shadow[k].to(sd[k].dtype) for k in sd})


class PAUCAtRecall(nn.Module):
    """recall=0.9 지점의 FPR 을 직접 줄이는 대리 손실.

        q  = 양성 점수의 10-분위수 (soft, torch.quantile 은 순서통계량으로 미분 가능)
        L  = mean_neg softplus((s_neg - q) / tau)

    -> q 위에 있는 음성만 벌점 (= FP), 동시에 q 를 만드는 하위 양성이 밀려 올라간다.
    양성이 배치당 적으므로 과거 양성 점수를 큐에 쌓아 분위수 추정을 안정화한다.
    """
    def __init__(self, target_recall=0.90, tau=1.0, queue=1024):
        super().__init__()
        self.tr, self.tau, self.qsize = target_recall, tau, queue
        self.register_buffer("q", torch.zeros(0))

    def forward(self, logits, y):
        pos, neg = logits[y > 0.5], logits[y < 0.5]
        if pos.numel() < 2 or neg.numel() < 1:
            return logits.sum() * 0.0
        allpos = torch.cat([self.q.to(logits.device, logits.dtype), pos]) if self.q.numel() else pos
        thr = torch.quantile(allpos.float(), 1.0 - self.tr).to(logits.dtype)
        loss = F.softplus((neg - thr) / self.tau).mean()
        with torch.no_grad():
            self.q = torch.cat([self.q.to(pos.device), pos.detach().float()])[-self.qsize:]
        return loss


# ======================================================================================
# 설정
# ======================================================================================
@dataclass
class Cfg:
    sampler: str = "center"       # none | class | center
    transplant_p: float = 0.5     # 0.0 이면 비활성
    mask_aware: bool = True
    pool: str = "topk"
    topk_frac: float = 0.02
    loss: str = "bce+pauc"        # bce | bce+pauc
    pauc_weight: float = 1.0
    pauc_tau: float = 1.0
    pauc_warmup: int = 5
    fov_crop: bool = True
    color_const: str = "none"     # none | sog | grayworld  (학습·추론 동일 적용)
    sharpness_p: float = 0.0      # 구조강조(블러/샤프닝) 증강 확률. 0.0 이면 비활성
    fourier_p: float = 0.0        # Fourier 진폭증강 확률 (도메인 랜덤화). 0.0 비활성
    fourier_L: float = 0.01       # 섭동할 저주파 대역 비율
    fourier_eta: float = 0.5      # 진폭 섭동 강도 (1±eta)


PRESETS = {
    # --- Step 4c: top-k 승자 위에서 재조합 ---
    # 승자 기준선 (Step 4b: LOCO fpr90 0.086, CI[0.062,0.113], plain 대비 유의)
    "topk":          Cfg(sampler="class", transplant_p=0.0, mask_aware=False, pool="topk", loss="bce"),
    # 꼬리를 살린 뒤 pAUC 로 순위 다듬기 (GAP 위에선 무익했으나 top-k 위에선 통할 수 있음)
    "topk_pauc":     Cfg(sampler="class", transplant_p=0.0, mask_aware=False, pool="topk", loss="bce+pauc"),
    # 구조강조 증강 프리셋: topk 와 sharpness_p 만 다름 (단일 변수).
    # 목적 (2가지 동시): ① 동등강도 앙상블 멤버 확보 (IMSY 식 다중 프리셋)
    #                    ② 장비 강조모드 차이 = 문서화된 실패모드 대응 (우리 사각지대)
    "topk_sharp":    Cfg(sampler="class", transplant_p=0.0, mask_aware=False, pool="topk",
                         loss="bce", sharpness_p=0.5),
    # top-k + 색항등성 (직교 축 — 더해질 가능성)
    "topk_cc":       Cfg(sampler="class", transplant_p=0.0, mask_aware=False, pool="topk", loss="bce",
                         color_const="sog"),
    # 셋 다
    "topk_cc_pauc":  Cfg(sampler="class", transplant_p=0.0, mask_aware=False, pool="topk", loss="bce+pauc",
                         color_const="sog"),
    # LSE pooling (top-k 의 부드러운 대안 — k 선택 불필요)
    "lse":           Cfg(sampler="class", transplant_p=0.0, mask_aware=False, pool="lse", loss="bce"),

    # --- Step 4b (일반화 절제) ---
    "plain":         Cfg(sampler="class", transplant_p=0.0, mask_aware=False, pool="gap", loss="bce"),
    "plain_pauc":    Cfg(sampler="class", transplant_p=0.0, mask_aware=False, pool="gap", loss="bce+pauc"),
    "plain_cc":      Cfg(sampler="class", transplant_p=0.0, mask_aware=False, pool="gap", loss="bce",
                         color_const="sog"),
    "plain_topk":    Cfg(sampler="class", transplant_p=0.0, mask_aware=False, pool="topk", loss="bce"),
    "plain_cc_pauc": Cfg(sampler="class", transplant_p=0.0, mask_aware=False, pool="gap", loss="bce+pauc",
                         color_const="sog"),

    # --- Step 4a archive (기각된 shortcut 처방) ---
    "naive":               Cfg(sampler="class", transplant_p=0.0, mask_aware=False, pool="gap", loss="bce"),
    "balance":             Cfg(sampler="center", transplant_p=0.0, mask_aware=False, pool="gap", loss="bce"),
    "balance_transplant":  Cfg(sampler="center", transplant_p=0.5, mask_aware=False, pool="gap", loss="bce"),
    "full":                Cfg(sampler="center", transplant_p=0.5, mask_aware=True, pool="topk", loss="bce+pauc"),
}


def make_sampler(df: pd.DataFrame, mode: str):
    if mode == "none":
        return None
    if mode == "class":
        key = df["label"].astype(str)
    else:  # center-conditional: {c1·ndbe, c1·neo, c2·ndbe, c2·neo} 를 균등하게
        if df["center"].nunique() < 2:
            print("  [!] 학습셋에 센터가 1개뿐 -> center-balanced 는 class-balanced 와 동일합니다.")
            print("      center shortcut 절제는 --split kfold 에서만 의미가 있습니다.")
        key = df["center"].astype(str) + "_" + df["label"].astype(str)
    cnt = key.value_counts()
    w = np.ascontiguousarray(key.map(lambda k: 1.0 / cnt[k]).to_numpy(np.float64))
    return WeightedRandomSampler(torch.tensor(w, dtype=torch.double), num_samples=len(df), replacement=True)


def load_stencils(bank: Path, n: int, size: int) -> Optional[np.ndarray]:
    if not bank or not bank.exists():
        return None
    files = sorted(bank.rglob("*.png"))
    if not files:
        return None
    random.Random(0).shuffle(files)
    out = []
    for f in files[:n]:
        m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        out.append((cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8))
    print(f"[stencils] {len(out)} masks, 평균 커버 {np.mean([s.mean() for s in out])*100:.2f}%")
    return np.stack(out) if out else None


# ======================================================================================
# 평가
# ======================================================================================
@torch.no_grad()
def predict(model, ds, device, bs, workers, variant_seed: Optional[int] = None,
            amp_dt=torch.bfloat16, amp_on=True):
    model.eval()
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers)
    S, Y = [], []
    for x, m, y, idx in dl:
        if variant_seed is not None:
            pairs = [ds.eval_variant(int(j), variant_seed) for j in idx]
            x = torch.stack([p[0] for p in pairs]); m = torch.stack([p[1] for p in pairs])
        x = x.to(device).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=amp_dt, enabled=amp_on):
            v = model(x, m.to(device))
        S.append(v.float().cpu()); Y.append(y)
    return torch.cat(S).numpy(), torch.cat(Y).numpy()


def report(y, s, tag=""):
    d = dict(pauc80=pauc_high_sens(y, s, 0.80), fpr90=fpr_at_tpr(y, s, 0.90),
             auroc=float(roc_auc_score(y, s)))
    d["ppv90"] = ppv_from_fpr(d["fpr90"])
    if tag:
        print(f"  {tag:<10} pauc80={d['pauc80']:.4f}  fpr90={d['fpr90']:.4f}  "
              f"ppv@90R={d['ppv90']:.4f}  auroc={d['auroc']:.4f}")
    return d


# ======================================================================================
# 학습
# ======================================================================================
def run_one(args, cfg: Cfg, tr: pd.DataFrame, va: pd.DataFrame, tag: str) -> Dict:
    dev = args.device
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    if dev == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"[gpu] {torch.cuda.get_device_name(0)}  amp={args.amp}")

    amp_dt = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp, torch.float32)
    amp_on = (args.amp != "off") and dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp == "fp16" and dev == "cuda"))

    bank = Path(args.mask_bank) if args.mask_bank else None
    need_st = (cfg.transplant_p > 0) or args.footprint_test
    stencils = load_stencils(bank, args.n_stencils, args.size) if need_st else None

    ds_tr = RareDS(tr, Path(args.data_root), bank, args.size, True, cfg,
                   stencils if cfg.transplant_p > 0 else None, args.cache)
    ds_va = RareDS(va, Path(args.data_root), bank, args.size, False, cfg,
                   stencils if args.footprint_test else None, args.cache)

    # ── 외부 학습 증강 (EDD2020 등): 별도 RareDS(bank=None, 절대경로) → ConcatDataset
    #    검증(ds_va)엔 절대 안 들어감. 마스크 없음 → load_pair 가 0 마스크 사용.
    sampler_df = tr
    if args.aug_manifest:
        aug_df = pd.read_csv(args.aug_manifest)
        # rel 절대경로 → root 는 pathlib 이 무시. bank=None → 마스크 로드 안 함.
        ds_aug = RareDS(aug_df, Path("/"), None, args.size, True, cfg,
                        stencils if cfg.transplant_p > 0 else None, args.cache)
        n_pos = int((aug_df["label"] == 1).sum()); n_neg = int((aug_df["label"] == 0).sum())
        print(f"[aug] {args.aug_manifest}: +{len(aug_df)}장 (양성 {n_pos}, 음성 {n_neg}) "
              f"학습셋에만 추가, 센터 {sorted(aug_df['center'].unique())}")
        ds_tr = ConcatDataset([ds_tr, ds_aug])
        # 샘플러는 결합 라벨/센터 순서 [tr..., aug...] 로 계산 (ConcatDataset 순서와 일치)
        sampler_df = pd.concat([tr[["label", "center"]], aug_df[["label", "center"]]],
                               ignore_index=True)

    dl = DataLoader(ds_tr, batch_size=args.bs, sampler=make_sampler(sampler_df, cfg.sampler),
                    shuffle=(cfg.sampler == "none"), num_workers=args.workers,
                    drop_last=True, pin_memory=True, persistent_workers=args.workers > 0)

    # 색불변 consistency: unlabeled 식도 프레임 (GastroNet DAPT)
    cons_iter = None
    if args.consist_dir:
        cfiles = [str(p) for p in sorted(Path(args.consist_dir).glob("*.png"))]
        if cfiles:
            cds = ConsistencyDS(cfiles, args.size, cfg.fov_crop)
            cbs = max(4, args.bs // 2)
            cdl = DataLoader(cds, batch_size=cbs, shuffle=True, num_workers=max(2, args.workers//2),
                             drop_last=True, pin_memory=True, persistent_workers=True)
            print(f"[consistency] 식도 unlabeled {len(cfiles)}장, bs={cbs}, "
                  f"weight={args.consist_weight}")
            def _cyc(dl_):
                while True:
                    for b in dl_:
                        yield b
            cons_iter = _cyc(cdl)
        else:
            print(f"[consistency] {args.consist_dir} 에 png 없음 -> consistency 미사용")

    model = Net(args.weights, cfg.pool, cfg.topk_frac, cfg.mask_aware,
                backbone=getattr(args, "backbone", "resnet50"),
                lora_r=getattr(args, "lora_r", 8)).to(dev)
    model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, epochs=args.epochs,
                                              steps_per_epoch=len(dl), pct_start=0.25)
    pauc = PAUCAtRecall(0.90, cfg.pauc_tau).to(dev)
    use_pauc = cfg.loss == "bce+pauc"
    if use_pauc and args.epochs <= cfg.pauc_warmup:
        print(f"  [!] epochs({args.epochs}) <= pauc_warmup({cfg.pauc_warmup}). "
              f"pAUC 손실이 한 번도 켜지지 않습니다.")

    ema = EMA(model, decay=args.ema_decay) if args.ema else None

    # EVC seg 보조: 별도 로더에서 배치를 당겨 seg loss 추가 (consistency 와 동일 패턴).
    # clean 학습 루프를 안 건드리고, EVC seg 신호만 병렬 주입.
    seg_iter = None
    if getattr(args, "seg_weight", 0.0) > 0 and getattr(args, "evc_root", None):
        from evc_seg_dataset import EvcSegDataset, seg_collate
        seg_ds = EvcSegDataset(args.evc_root, size=args.size,
                               color_const=cfg.color_const, train=True)
        seg_bs = max(2, args.bs // 2)
        seg_loader = DataLoader(seg_ds, batch_size=seg_bs, shuffle=True,
                                num_workers=max(2, args.workers // 2), drop_last=True,
                                collate_fn=seg_collate, pin_memory=True)

        def _seg_cycle(loader):
            while True:
                for b in loader:
                    yield b
        seg_iter = _seg_cycle(seg_loader)
        print(f"[seg] EVC multi-task 활성: seg_weight={args.seg_weight} bs={seg_bs} "
              f"warmup={args.seg_warmup}")

    def dice_bce_loss(logit_map, soft_target):
        """seg 손실 = BCE(soft) + Dice(soft). logit_map:(B,1,h,w), soft_target:(B,1,H,W)."""
        B, _, h, w = logit_map.shape
        tgt = F.interpolate(soft_target, size=(h, w), mode="bilinear", align_corners=False)
        prob = torch.sigmoid(logit_map)
        bce = F.binary_cross_entropy_with_logits(logit_map, tgt)
        inter = (prob * tgt).flatten(1).sum(1)
        denom = prob.flatten(1).sum(1) + tgt.flatten(1).sum(1)
        dice = 1.0 - (2 * inter + 1.0) / (denom + 1.0)
        return bce + dice.mean()

    best = {"pauc80": -1}
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0.0; npauc = 0; tcons = 0.0; tseg = 0.0
        cw = args.consist_weight if (cons_iter is not None and ep >= args.consist_warmup) else 0.0
        sw = args.seg_weight if (seg_iter is not None and ep >= args.seg_warmup) else 0.0
        for x, m, y, _ in dl:
            x = x.to(dev, non_blocking=True).contiguous(memory_format=torch.channels_last)
            m, y = m.to(dev, non_blocking=True), y.to(dev)
            with torch.autocast("cuda", dtype=amp_dt, enabled=amp_on):
                v = model(x, m)
                loss = F.binary_cross_entropy_with_logits(v, y)
                if use_pauc and ep >= cfg.pauc_warmup:
                    loss = loss + cfg.pauc_weight * pauc(v, y); npauc += 1
                # 색불변 consistency: 같은 형태·다른 색 두 뷰가 같은 logit
                if cw > 0:
                    v1v, v2v, zm = next(cons_iter)
                    v1v = v1v.to(dev, non_blocking=True).contiguous(memory_format=torch.channels_last)
                    v2v = v2v.to(dev, non_blocking=True).contiguous(memory_format=torch.channels_last)
                    zm = zm.to(dev, non_blocking=True)
                    l1 = model(v1v, zm); l2 = model(v2v, zm)
                    closs = F.mse_loss(torch.sigmoid(l1), torch.sigmoid(l2))
                    loss = loss + cw * closs
                    tcons += float(closs)
                # EVC seg 보조: 병변 위치 supervision 으로 국소 표현 강화
                if sw > 0:
                    sx, sm, sy, sseg, shas, _ = next(seg_iter)
                    sx = sx.to(dev, non_blocking=True).contiguous(memory_format=torch.channels_last)
                    sm = sm.to(dev, non_blocking=True)
                    sy = sy.to(dev, non_blocking=True)
                    sseg = sseg.to(dev, non_blocking=True)
                    scls, slm = model(sx, sm, return_map=True)   # 분류 logit + seg map
                    # EVC 도 분류 학습(cls) + seg 학습(seg)
                    seg_cls = F.binary_cross_entropy_with_logits(scls, sy)
                    seg_loss = dice_bce_loss(slm, sseg)
                    loss = loss + sw * seg_loss + 0.5 * sw * seg_cls
                    tseg += float(seg_loss)
            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            sch.step(); tot += float(loss)
            if ema is not None:
                ema.update(model)

        last = (ep == args.epochs - 1)
        if not (last or (ep + 1) % args.eval_every == 0):
            cmsg = f" cons={tcons/len(dl):.4f}" if cw > 0 else ""
            smsg = f" seg={tseg/len(dl):.4f}" if sw > 0 else ""
            print(f"  ep{ep:02d} loss={tot/len(dl):.4f}{cmsg}{smsg} ({time.time()-t0:.0f}s)")
            continue

        # raw 가중치 평가
        s, y = predict(model, ds_va, dev, args.bs, args.workers, amp_dt=amp_dt, amp_on=amp_on)
        d = report(y, s)
        line = (f"  ep{ep:02d} loss={tot/len(dl):.4f} pauc80={d['pauc80']:.4f} "
                f"fpr90={d['fpr90']:.4f} ppv={d['ppv90']:.4f} auroc={d['auroc']:.4f}")

        # EMA 가중치 평가 (있으면 이쪽을 우선 선택 — 더 안정적)
        cand, cand_s, cand_y, src = d, s, y, "raw"
        if ema is not None and ep >= max(1, args.epochs // 3):
            backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.copy_to(model)
            se, ye = predict(model, ds_va, dev, args.bs, args.workers, amp_dt=amp_dt, amp_on=amp_on)
            de = report(ye, se)
            model.load_state_dict(backup)
            line += f" | ema pauc80={de['pauc80']:.4f} fpr90={de['fpr90']:.4f}"
            if de["pauc80"] >= d["pauc80"]:
                cand, cand_s, cand_y, src = de, se, ye, "ema"
        line += f" {'[pauc]' if npauc else ''} ({time.time()-t0:.0f}s)"
        print(line)

        if cand["pauc80"] > best["pauc80"]:
            best = {**cand, "epoch": ep, "src": src, "scores": cand_s, "y": cand_y}
            # 선택된 가중치를 저장 (footprint-test 재현용)
            if src == "ema":
                backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
                ema.copy_to(model)
                torch.save(model.state_dict(), Path(args.out_dir) / tag / "best.pt")
                model.load_state_dict(backup)
            else:
                torch.save(model.state_dict(), Path(args.out_dir) / tag / "best.pt")

    out = Path(args.out_dir) / tag
    pd.DataFrame({"filename": va.rel.values, "label": best["y"].astype(int),
                  "score": 1 / (1 + np.exp(-best["scores"])), "logit": best["scores"],
                  "center": va.center.values}).to_csv(out / "preds.csv", index=False)

    print(f"\n  [BEST ep{best['epoch']}]")
    report(best["y"], best["scores"], "best")
    ci = boot_ci(best["y"], best["scores"])
    print(f"  95% CI   fpr90 [{ci['fpr90_lo']:.4f}, {ci['fpr90_hi']:.4f}]"
          f"   -> ppv@90R [{ppv_from_fpr(ci['fpr90_hi']):.4f}, {ppv_from_fpr(ci['fpr90_lo']):.4f}]")
    print(f"           pauc80 [{ci['pauc80_lo']:.4f}, {ci['pauc80_hi']:.4f}]")
    print(f"  [주의] arm 간 fpr90 차이가 CI 폭({ci['fpr90_hi']-ci['fpr90_lo']:.4f})보다 작으면 '차이 없음' 입니다.")

    # ---------------- shortcut 진단 ----------------
    diag = dict(ci)
    if va.center.nunique() > 1:
        nd = best["y"] == 0
        c = (va.center.values[nd] == sorted(va.center.unique())[0]).astype(int)
        if len(np.unique(c)) == 2:
            diag["auc_score_to_center_on_ndbe"] = float(roc_auc_score(c, best["scores"][nd]))
            print(f"  [진단] AUC(score -> center | NDBE) = {diag['auc_score_to_center_on_ndbe']:.3f}"
                  f"   (0.5 이면 shortcut 없음)")
    else:
        print("  [진단] val 이 단일 센터 -> AUC(score->center) 계산 불가. shortcut 절제는 kfold 로.")

    if args.footprint_test and ds_va.stencils is not None:
        model.load_state_dict(torch.load(out / "best.pt", map_location=dev))
        s2, _ = predict(model, ds_va, dev, args.bs, args.workers, variant_seed=1234,
                        amp_dt=amp_dt, amp_on=amp_on)
        d2 = report(best["y"], s2, "w/ stencil")
        dlt = np.abs(s2 - best["scores"])
        diag.update(footprint_mean_abs_dlogit=float(dlt.mean()),
                    footprint_p95_abs_dlogit=float(np.percentile(dlt, 95)),
                    fpr90_with_stencil=d2["fpr90"], fpr90_delta=d2["fpr90"] - best["fpr90"])
        print(f"  [진단] footprint 이식 시 |Δlogit| 평균 {dlt.mean():.3f} (p95 {np.percentile(dlt,95):.3f})")
        print(f"         fpr90 {best['fpr90']:.4f} -> {d2['fpr90']:.4f}  "
              f"(Δ{d2['fpr90']-best['fpr90']:+.4f})   0 에 가까워야 정상")

    res = {"tag": tag, "cfg": asdict(cfg), "epoch": best["epoch"],
           **{k: best[k] for k in ("pauc80", "fpr90", "ppv90", "auroc")}, **diag,
           "n_train": len(tr), "n_val": len(va), "pos_val": int(best["y"].sum())}
    (out / "result.json").write_text(json.dumps(res, indent=2))
    return res


def summarize(root: str):
    rows = [json.loads(p.read_text()) for p in sorted(Path(root).rglob("result.json"))]
    if not rows:
        print("result.json 없음"); return
    df = pd.DataFrame(rows)
    cols = [c for c in ["tag", "pauc80", "fpr90", "ppv90", "auroc",
                        "fpr90_lo", "fpr90_hi", "pauc80_lo", "pauc80_hi",
                        "auc_score_to_center_on_ndbe", "footprint_mean_abs_dlogit",
                        "fpr90_delta", "pos_val", "src"] if c in df.columns]
    print(df[cols].sort_values("fpr90").to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # preset 별 OOF 통합 (kfold 런들을 preset 으로 묶어 158 양성 전체에서 재계산)
    oof_files = list(Path(root).rglob("*_kfold*/preds.csv"))
    if oof_files:
        print("\n" + "─" * 60 + "\n[OOF] preset 별 5-fold 통합 (tail=16장, CI 좁음)")
        import re
        buckets: Dict[str, List[pd.DataFrame]] = {}
        for f in oof_files:
            m = re.match(r"(.+)_kfold\d+$", f.parent.name)
            if m:
                buckets.setdefault(m.group(1), []).append(pd.read_csv(f))
        for preset, dfs in sorted(buckets.items()):
            o = pd.concat(dfs)
            y, s = o.label.values, o.score.values
            fpr = fpr_at_tpr(y, s); pa = pauc_high_sens(y, s)
            ci = boot_ci(y, s, n=1000)
            print(f"  {preset:<20} n_folds={len(dfs)} pos={int(y.sum())}  "
                  f"fpr90={fpr:.4f} [{ci['fpr90_lo']:.4f},{ci['fpr90_hi']:.4f}]  "
                  f"ppv@90R={ppv_from_fpr(fpr):.4f}  pauc80={pa:.4f}  auroc={roc_auc_score(y,s):.4f}")
        print("  * [주의] k-fold 는 학습·검증에 같은 2센터가 있어 '센터 prior' 로 부풀려집니다.")
        print("           테스트(미지 센터) 예측력은 LOCO 표를 보세요.")

    # preset 별 LOCO: 방향을 분리한다.
    #   c1->c2 : eval 양성 97 -> tail ~10장, 측정 가능. ★ primary 프록시
    #   c2->c1 : eval 양성 61 -> tail ~6장, EMA/epoch 선택에 뒤집힘. 참고용.
    loco_files = list(Path(root).rglob("*_loco_*/preds.csv"))
    if loco_files:
        import re
        print("\n" + "─" * 70)
        print("[LOCO] 방향별 분리  ★ 'train_c1' (eval c2, 양성 97) 열만 신뢰하세요")
        print("        'train_c2' (eval c1, 양성 61, tail 6장) 은 노이즈가 지배합니다.")
        by_dir: Dict[str, Dict[str, list]] = {"center_1": {}, "center_2": {}}
        for f in loco_files:
            m = re.match(r"(.+)_loco_(center_\d+)(_s\d+)?$", f.parent.name)
            if not m:
                continue
            preset, tr_center = m.group(1), m.group(2)
            o = pd.read_csv(f)
            y, s = o.label.values, o.score.values
            if len(np.unique(y)) < 2:
                continue
            fpr = fpr_at_tpr(y, s); pa = pauc_high_sens(y, s)
            by_dir[tr_center].setdefault(preset, []).append((fpr, pa, int(y.sum())))

        presets = sorted(set(list(by_dir["center_1"]) + list(by_dir["center_2"])))

        def agg(center, p):
            """여러 시드가 있으면 fpr90 median 과 (min,max) 범위를 반환."""
            v = by_dir[center].get(p)
            if not v:
                return None
            f = np.array([x[0] for x in v]); a = np.array([x[1] for x in v])
            return (float(np.median(f)), float(f.min()), float(f.max()),
                    float(np.median(a)), len(v))

        print(f"  {'preset':<15}{'train_c1 fpr90 (med[min..max]) pauc':<42}{'train_c2 fpr90 (참고)':<26}")
        def keyf(p):
            r = agg('center_1', p)
            return r[0] if r else 9
        best = min((keyf(p) for p in presets), default=9)
        for p in sorted(presets, key=keyf):
            a = agg('center_1', p); b = agg('center_2', p)
            sa = f"{a[0]:.4f} [{a[1]:.3f}..{a[2]:.3f}] {a[3]:.3f} (n{a[4]})" if a else "—"
            sb = f"{b[0]:.4f} [{b[1]:.3f}..{b[2]:.3f}]" if b else "—"
            star = "  <<<" if a and abs(a[0] - best) < 1e-9 else ""
            print(f"  {p:<15}{sa:<42}{sb:<26}{star}")
        print("\n  * train_c1 (eval c2, 양성 97) fpr90 median 최소 arm 채택.")
        print("    [min..max] 범위가 넓으면 시드 불안정 -> 재현성 의심. 다중 시드 권장.")
        print("  * PPV@90R = 0.9/(0.9+100*fpr90). 이 값도 '단일' 미지 센터 프록시임에 유의.")

    print("\n해석:")
    print("  fpr90 낮을수록 좋음.  PPV@90R = 0.9/(0.9+100*fpr90)")
    print("  auc_score_to_center_on_ndbe -> 0.5 여야 shortcut 없음")
    print("  fpr90_delta -> 0 이어야 footprint 에 불변")


# ======================================================================================
# main
# ======================================================================================
def load_extra_train(csv_path, root, base_cols):
    """외부 학습 데이터를 folds 스키마로 변환. train 에만 추가된다 (검증 오염 방지).
    CSV 필수 컬럼: img_path(또는 path/rel), label.
    center='extra', fold=-99 (어느 k 검증에도 안 걸림), group_id 음수 (그룹 충돌 없음).
    """
    ex = pd.read_csv(csv_path)
    pcol = next((c for c in ("img_path", "path", "rel") if c in ex.columns), None)
    if pcol is None or "label" not in ex.columns:
        raise ValueError(f"extra CSV 에 img_path/path/rel 중 하나와 label 필요. 컬럼: {list(ex.columns)}")

    def mkabs(p):
        p = str(p)
        return p if p.startswith("/") else (str(Path(root) / p) if root else p)

    out = pd.DataFrame()
    out["rel"] = ex[pcol].map(mkabs)
    out["filename"] = out["rel"].map(lambda p: Path(p).name)
    out["center"] = "extra"
    out["label"] = ex["label"].astype(int)
    out["fold"] = -99
    out["group_id"] = -(np.arange(len(out)) + 1)
    for c in base_cols:
        if c not in out.columns:
            out[c] = np.nan
    missing = [f for f in out["rel"] if not Path(f).exists()]
    if missing:
        print(f"  [!] extra 이미지 {len(missing)}개 경로 없음 (예: {missing[0]})")
    return out[base_cols]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize", default=None, help="runs 디렉토리 요약 후 종료")
    ap.add_argument("--data-root"); ap.add_argument("--folds")
    ap.add_argument("--mask-bank", default=None)
    ap.add_argument("--aug-manifest", default=None,
                    help="외부 학습 증강 CSV (rel 절대경로, label, center). "
                         "학습셋에만 추가되고 검증엔 안 들어감. 마스크 없음(bank=None) 처리.")
    ap.add_argument("--weights", default="none")
    ap.add_argument("--backbone", default="resnet50",
                    choices=["resnet50", "convnext_base", "vits_gastronet", "dinov3_vitl",
                             "maxvit_tiny", "caformer_s18"],
                    help="백본 몸통. convnext_base 는 --weights 에 timm 태그(fb_in22k) 또는 none")
    ap.add_argument("--out-dir", default="./runs")
    ap.add_argument("--preset", default="full", choices=list(PRESETS))
    ap.add_argument("--color-const", default=None,
                    choices=["none", "sog", "grayworld", "wb", "clahe", "wb_clahe"],
                    help="preset 의 색항등성 설정을 덮어씀")
    ap.add_argument("--split", default="loco", choices=["loco", "kfold"])
    ap.add_argument("--loco-train", default="center_1")
    ap.add_argument("--fold", default="0", help="kfold: 0..4 또는 all")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-stencils", type=int, default=300)
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"],
                    help="Turing(RTX 6000) 은 bf16 네이티브 미지원 -> fp16 권장")
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--ema", action="store_true", help="가중치 EMA (epoch 변동 억제)")
    ap.add_argument("--ema-decay", type=float, default=0.997)
    ap.add_argument("--require-gpu", default=None,
                    help="GPU 이름에 이 문자열이 없으면 즉시 중단 (예: Blackwell, RTX PRO)")
    ap.add_argument("--pauc-warmup", type=int, default=None)
    ap.add_argument("--pauc-weight", type=float, default=None)
    ap.add_argument("--pauc-tau", type=float, default=None)
    ap.add_argument("--topk-frac", type=float, default=None)
    ap.add_argument("--lora-r", type=int, default=8,
                    help="LoRA rank (dinov3_vitl 용). 8 이면 ~0.8M 학습")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--extra-train", default=None,
                    help="외부 학습 데이터 CSV (rel/절대경로 img_path, label, [weight]). "
                         "train 에만 추가되고 검증에는 절대 안 들어감. hard-neg 투입용.")
    ap.add_argument("--extra-root", default=None,
                    help="extra CSV 의 상대경로 기준 루트 (절대경로면 불필요)")
    ap.add_argument("--consist-dir", default=None,
                    help="색불변 consistency 용 unlabeled 식도 프레임 폴더 (GastroNet DAPT)")
    ap.add_argument("--consist-weight", type=float, default=1.0,
                    help="consistency loss 가중치 (기본 1.0)")
    ap.add_argument("--consist-warmup", type=int, default=2,
                    help="이 epoch 부터 consistency 켬 (초기 안정화)")
    ap.add_argument("--fourier-p", type=float, default=None,
                    help="Fourier 진폭증강 확률 (도메인 랜덤화). CLAHE 넘어 색/스캐너 갭 공격")
    ap.add_argument("--fourier-L", type=float, default=None,
                    help="섭동할 저주파 대역 비율 (기본 0.01)")
    ap.add_argument("--fourier-eta", type=float, default=None,
                    help="진폭 섭동 강도 1±eta (기본 0.5)")
    ap.add_argument("--seg-weight", type=float, default=0.0,
                    help="EVC multi-task seg 손실 가중치. 0 이면 비활성. 국소 병변 표현 강화")
    ap.add_argument("--seg-warmup", type=int, default=3,
                    help="seg 손실 시작 epoch (cls 안정화 후)")
    ap.add_argument("--evc-root", default=None,
                    help="EVC segmentation 데이터 루트 (seg-weight>0 시 필요)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cache", dest="cache", action="store_false")
    ap.add_argument("--footprint-test", action="store_true")
    args = ap.parse_args(argv)

    if args.summarize:
        return summarize(args.summarize)
    if not (args.data_root and args.folds):
        ap.error("--data-root 와 --folds 가 필요합니다.")

    cfg = PRESETS[args.preset]
    if args.color_const is not None:
        cfg.color_const = args.color_const
    for k in ("pauc_warmup", "pauc_weight", "pauc_tau", "topk_frac"):
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)
    for k in ("fourier_p", "fourier_L", "fourier_eta"):
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)
    df = pd.read_csv(args.folds)

    extra = None
    if args.extra_train:
        extra = load_extra_train(args.extra_train, args.extra_root, list(df.columns))
        print(f"[extra] {len(extra)}장 (label 분포: {dict(extra.label.value_counts())}) "
              f"-> train 에만 추가, 검증 제외")

    if args.device == "cuda":
        name = torch.cuda.get_device_name(0)
        print(f"[gpu] visible device 0 = {name}")
        if args.require_gpu and args.require_gpu.lower() not in name.lower():
            print(f"[STOP] '{args.require_gpu}' 를 요구했으나 '{name}' 입니다. "
                  f"CUDA_VISIBLE_DEVICES 를 확인하세요.")
            return 1
        cap = torch.cuda.get_device_capability(0)
        if args.amp == "bf16" and cap[0] < 8:
            print(f"[!] compute capability {cap} 은 bf16 텐서코어 미지원. --amp fp16 을 권장합니다.")
    print(f"[cfg] {args.preset}: {asdict(cfg)}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    results = []

    if args.split == "loco":
        tr = df[df.center == args.loco_train]
        va = df[df.center != args.loco_train]
        if extra is not None:
            tr = pd.concat([tr, extra], ignore_index=True)
        sfx = f"_s{args.seed}" if args.seed != 0 else ""
        tag = f"{args.preset}_loco_{args.loco_train}{sfx}"
        (Path(args.out_dir) / tag).mkdir(parents=True, exist_ok=True)
        print(f"\n=== {tag}: train {len(tr)} (pos {int(tr.label.sum())}) "
              f"-> val {len(va)} (pos {int(va.label.sum())}) ===")
        results.append(run_one(args, cfg, tr, va, tag))
    else:
        folds = range(5) if args.fold == "all" else [int(args.fold)]
        oof = []
        for k in folds:
            tr, va = df[df.fold != k], df[df.fold == k]
            if extra is not None:
                tr = pd.concat([tr, extra], ignore_index=True)
            tag = f"{args.preset}_kfold{k}"
            (Path(args.out_dir) / tag).mkdir(parents=True, exist_ok=True)
            print(f"\n=== {tag}: train {len(tr)} -> val {len(va)} (pos {int(va.label.sum())}) ===")
            results.append(run_one(args, cfg, tr, va, tag))
            oof.append(pd.read_csv(Path(args.out_dir) / tag / "preds.csv"))
        if len(oof) == 5:
            o = pd.concat(oof)
            od = Path(args.out_dir) / f"{args.preset}_kfold_oof"; od.mkdir(exist_ok=True)
            o.to_csv(od / "preds.csv", index=False)
            print(f"\n=== OOF (158 양성 전체, tail=16장) ===")
            report(o.label.values, o.score.values, "OOF")
            print(f"[saved] {od/'preds.csv'}  ->  python rare_metrics.py --csv {od/'preds.csv'} --group-col center")

    print("\n" + "=" * 92)
    for r in results:
        print(f"RESULT | {r['tag']:<28} pauc80={r['pauc80']:.4f}  fpr90={r['fpr90']:.4f}  "
              f"ppv@90R={r['ppv90']:.4f}  auroc={r['auroc']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
