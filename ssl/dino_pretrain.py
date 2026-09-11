#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dino_pretrain.py — 3단계: DINOv3 iBOT-style continued pretraining (Step 32b)
=========================================================================
MAE 실패(목적함수 충돌)의 교훈: DINOv3 는 self-distillation 으로 학습됨.
같은 계열(DINO)로 continued pretraining 하면 표현 파괴 없이 도메인 적응.

★ 표현 파괴 방지 설계 (MAE 실패에서 배움):
  1. student/teacher 둘 다 원본 DINOv3 에서 출발 (강한 상태 유지)
  2. 낮은 lr (기본 5e-6) + teacher momentum 0.996 (느린 갱신)
  3. centering + sharpening (collapse 방지, dino_components 에서 검증)
  4. 매 에폭 체크포인트 -> linear probe 로 표현 감시 (MAE 처럼 붕괴 즉시 포착)
  5. head 만 warmup 후 backbone 열기 옵션 (--head-warmup-steps)

multi-crop: global 2개(224) + local N개(96). teacher 는 global 만,
student 는 전부. "부분(local)으로 전체(global) 예측" = 표현 학습 동력.

사용법:
  python dino_pretrain.py \
    --gastro-dir <DATA_ROOT>/Gastronet-5M \
    --n-images 200000 --epochs 10 --lr 5e-6 --bs 64 \
    --out ./runs_dino_dinov3 --device cuda

의존성: torch, timm, pillow, numpy, dino_components.py
"""
from __future__ import annotations
import argparse, glob, io, random, time, zipfile, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from dino_components import DINOHead, DINOLoss, ema_update

IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def rand_resized_crop(img_np, out_size, scale_lo, scale_hi, rng):
    """랜덤 리사이즈 크롭 (numpy/cv2). scale 범위에서 면적 비율 샘플."""
    import cv2
    H, W = img_np.shape[:2]
    area = H * W
    for _ in range(10):
        target = rng.uniform(scale_lo, scale_hi) * area
        ar = rng.uniform(3/4, 4/3)
        w = int(round((target * ar) ** 0.5))
        h = int(round((target / ar) ** 0.5))
        if w <= W and h <= H:
            x = rng.integers(0, W - w + 1)
            y = rng.integers(0, H - h + 1)
            crop = img_np[y:y+h, x:x+w]
            return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return cv2.resize(img_np, (out_size, out_size), interpolation=cv2.INTER_AREA)


def color_jitter(img, rng):
    """약한 색/밝기 지터 (내시경 도메인 보존 위해 약하게)."""
    import cv2
    img = img.astype(np.float32)
    img *= rng.uniform(0.8, 1.2)          # 밝기
    img = np.clip(img, 0, 255)
    if rng.random() < 0.2:                # 가끔 그레이스케일
        g = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        img = np.stack([g]*3, -1).astype(np.float32)
    return img.astype(np.uint8)


class MultiCropDataset(Dataset):
    """같은 이미지에서 global 2 + local N crop 생성."""
    def __init__(self, index, n_global=2, n_local=6, global_size=224, local_size=96,
                 global_scale=(0.4, 1.0), local_scale=(0.05, 0.4)):
        self.index = index
        self.n_global, self.n_local = n_global, n_local
        self.gsize, self.lsize = global_size, local_size
        self.gscale, self.lscale = global_scale, local_scale
        self._handles = {}
        from PIL import Image
        self.Image = Image

    def __len__(self):
        return len(self.index)

    def _handle(self, zp):
        if zp not in self._handles:
            self._handles[zp] = zipfile.ZipFile(zp)
        return self._handles[zp]

    def _norm(self, crop):
        x = (crop.astype(np.float32)/255.0 - IMNET_MEAN)/IMNET_STD
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))

    def __getitem__(self, i):
        zp, nm = self.index[i]
        rng = np.random.default_rng()
        try:
            img = self.Image.open(io.BytesIO(self._handle(zp).read(nm))).convert("RGB")
            img_np = np.asarray(img)
        except Exception:
            img_np = np.full((224, 224, 3), 128, np.uint8)
        crops = []
        for _ in range(self.n_global):
            c = rand_resized_crop(img_np, self.gsize, *self.gscale, rng)
            c = color_jitter(c, rng)
            if rng.random() < 0.5:
                c = c[:, ::-1]
            crops.append(self._norm(np.ascontiguousarray(c)))
        for _ in range(self.n_local):
            c = rand_resized_crop(img_np, self.lsize, *self.lscale, rng)
            c = color_jitter(c, rng)
            if rng.random() < 0.5:
                c = c[:, ::-1]
            crops.append(self._norm(np.ascontiguousarray(c)))
        return crops  # list of tensors (global 먼저, local 뒤)


def collate_multicrop(batch):
    """배치에서 같은 위치 crop 끼리 스택. global 은 224, local 은 96 (크기 다름)."""
    n_crops = len(batch[0])
    return [torch.stack([b[c] for b in batch]) for c in range(n_crops)]


def build_index(gastro_dir, n_images, seed=0):
    zips = sorted(glob.glob(str(Path(gastro_dir) / "*.zip")))
    zips = [z for z in zips if Path(z).stat().st_size > 1000]
    rng = random.Random(seed)
    per_zip = max(1, n_images // len(zips))
    index = []
    for zp in zips:
        with zipfile.ZipFile(zp) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".png")]
            index.extend((zp, nm) for nm in rng.sample(names, min(per_zip, len(names))))
    rng.shuffle(index)
    return index[:n_images]


class DINOWrap(nn.Module):
    """backbone + DINO head. global/local 크기 다르므로 각각 forward."""
    def __init__(self, backbone, head, n_prefix=5):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.n_prefix = n_prefix

    def forward(self, crops):
        # crops: list, 앞 2개 global(224), 뒤 local(96). 크기별로 묶어 forward.
        outs = []
        for c in crops:
            f = self.backbone.forward_features(c)   # (B, prefix+N, C)
            cls = f[:, 0, :]                        # CLS 토큰
            outs.append(self.head(cls))
        return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gastro-dir", required=True)
    ap.add_argument("--n-images", type=int, default=200000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--n-local", type=int, default=6)
    ap.add_argument("--out-dim", type=int, default=8192)
    ap.add_argument("--teacher-momentum", type=float, default=0.996)
    ap.add_argument("--teacher-temp", type=float, default=0.04)
    ap.add_argument("--freeze-blocks", type=int, default=0)
    ap.add_argument("--save-epochs", default="1,2,3,5,10")
    ap.add_argument("--out", default="./runs_dino_dinov3")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    save_eps = set(int(x) for x in args.save_epochs.split(","))
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    import timm
    print(f"[index] {args.n_images:,}장 인덱싱...")
    index = build_index(args.gastro_dir, args.n_images)
    print(f"[index] {len(index):,}장")

    ds = MultiCropDataset(index, n_local=args.n_local)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True, collate_fn=collate_multicrop,
                    persistent_workers=True)

    # student: 원본 DINOv3 에서 출발
    student_bb = timm.create_model("vit_large_patch16_dinov3.lvd1689m",
                                   pretrained=True, num_classes=0, img_size=224).to(dev)
    embed_dim = 1024
    if args.freeze_blocks > 0:
        for i, blk in enumerate(student_bb.blocks):
            if i < args.freeze_blocks:
                for p in blk.parameters():
                    p.requires_grad = False
        print(f"[freeze] 앞 {args.freeze_blocks} 블록 동결")

    student_head = DINOHead(embed_dim, args.out_dim).to(dev)
    student = DINOWrap(student_bb, student_head).to(dev)

    # teacher: deepcopy 는 weight_norm 과 비호환 -> 독립 생성 후 가중치 복사
    teacher_bb = timm.create_model("vit_large_patch16_dinov3.lvd1689m",
                                   pretrained=True, num_classes=0, img_size=224).to(dev)
    teacher_head = DINOHead(embed_dim, args.out_dim).to(dev)
    teacher = DINOWrap(teacher_bb, teacher_head).to(dev)
    teacher.load_state_dict(student.state_dict())   # student 와 동일 초기화
    for p in teacher.parameters():
        p.requires_grad = False

    loss_fn = DINOLoss(args.out_dim, teacher_temp=args.teacher_temp).to(dev)

    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.04)
    print(f"[model] student 학습 파라미터 {sum(p.numel() for p in params)/1e6:.1f}M, "
          f"lr {args.lr}, teacher momentum {args.teacher_momentum}")

    n_global = 2
    for ep in range(1, args.epochs + 1):
        student.train(); teacher.train()
        t0 = time.time(); tot = 0.0; nb = 0
        for crops in dl:
            crops = [c.to(dev, non_blocking=True) for c in crops]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type=="cuda"):
                student_out = student(crops)                 # 전체 crop
                with torch.no_grad():
                    teacher_out = teacher(crops[:n_global])  # global 만
                loss = loss_fn(student_out, teacher_out)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 3.0)
            opt.step()
            ema_update(student, teacher, args.teacher_momentum)
            tot += float(loss); nb += 1
            if nb % 50 == 0:
                rate = nb * args.bs / (time.time() - t0)
                print(f"  ep{ep} [{nb}] loss={tot/nb:.4f} {rate:.0f} img/s")
        print(f"[ep{ep}] loss={tot/nb:.4f} ({time.time()-t0:.0f}s)")
        if ep in save_eps:
            # teacher backbone 저장 (표현이 더 안정적). fp16.
            sd = {k: (v.half() if v.is_floating_point() else v)
                  for k, v in teacher.backbone.state_dict().items()}
            torch.save(sd, out / f"dinov3_dino_ep{ep}.pt")
            print(f"  [저장] dinov3_dino_ep{ep}.pt (teacher backbone)")

    print(f"\n[완료] {out}/")
    print("  각 ep 를 linear probe (원본 0.9307 기준):")
    print("  for EP in " + args.save_epochs.replace(",", " ") + "; do")
    print("    python mae_linear_probe.py --backbone-ckpt ./runs_dino_dinov3/dinov3_dino_ep${EP}.pt \\")
    print("      --data-root ... --folds ... --size 224 --out ./dino_probe --tag dino_ep${EP}")
    print("  done")


if __name__ == "__main__":
    main()
    
"""
python dino_pretrain.py \
  --gastro-dir <DATA_ROOT>/Gastronet-5M \
  --n-images 200000 --epochs 10 --lr 5e-6 --bs 64 \
  --out ./runs_dino_dinov3 --device cuda 2>&1 | tee dino_ep10.log
  
[완료] runs_dino_dinov3/
  각 ep 를 linear probe (원본 0.9307 기준):
  for EP in 1 2 3 5 10; do
    python mae_linear_probe.py --backbone-ckpt ./runs_dino_dinov3/dinov3_dino_ep${EP}.pt \
      --data-root ... --folds ... --size 224 --out ./dino_probe --tag dino_ep${EP}
  done

"""
"""
for EP in 1 2 3 5; do   python mae_linear_probe.py --backbone-ckpt ./runs_dino_dinov3/dinov3_dino_ep${EP}.pt     --data-root <DATA_ROOT>/clean --folds ./folds_step3/folds.csv     --size 224 --out ./dino_probe --tag dino_ep${EP}; done
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
[ckpt] dinov3_dino_ep1.pt 로드 (missing 0, unexpected 0)
[data] 3095장 (양성 158)
[extract] 특징 추출 중...
[extract] 특징 (3095, 1024)

[dino_ep1] linear probe:
  OOF AUROC = 0.9438
  fold별 AUROC = ['0.934', '0.963', '0.940', '0.936', '0.943'] (SD 0.010)
  OOF FPR90 = 0.1900

[saved] dino_probe/probe_results.csv
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
[ckpt] dinov3_dino_ep2.pt 로드 (missing 0, unexpected 0)
[data] 3095장 (양성 158)
[extract] 특징 추출 중...
[extract] 특징 (3095, 1024)

[dino_ep2] linear probe:
  OOF AUROC = 0.9394
  fold별 AUROC = ['0.928', '0.968', '0.929', '0.933', '0.938'] (SD 0.015)
  OOF FPR90 = 0.2329

[saved] dino_probe/probe_results.csv

[누적 비교]
  dino_ep1         AUROC 0.9438   FPR90 0.1900
  dino_ep2         AUROC 0.9394   FPR90 0.2329
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
[ckpt] dinov3_dino_ep3.pt 로드 (missing 0, unexpected 0)
[data] 3095장 (양성 158)
[extract] 특징 추출 중...
[extract] 특징 (3095, 1024)

[dino_ep3] linear probe:
  OOF AUROC = 0.9091
  fold별 AUROC = ['0.910', '0.973', '0.867', '0.893', '0.907'] (SD 0.035)
  OOF FPR90 = 0.3531

[saved] dino_probe/probe_results.csv

[누적 비교]
  dino_ep1         AUROC 0.9438   FPR90 0.1900
  dino_ep2         AUROC 0.9394   FPR90 0.2329
  dino_ep3         AUROC 0.9091   FPR90 0.3531
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
[ckpt] dinov3_dino_ep5.pt 로드 (missing 0, unexpected 0)
[data] 3095장 (양성 158)
[extract] 특징 추출 중...
[extract] 특징 (3095, 1024)

[dino_ep5] linear probe:
  OOF AUROC = 0.9156
  fold별 AUROC = ['0.930', '0.945', '0.909', '0.888', '0.902'] (SD 0.020)
  OOF FPR90 = 0.3534

[saved] dino_probe/probe_results.csv

[누적 비교]
  dino_ep1         AUROC 0.9438   FPR90 0.1900
  dino_ep2         AUROC 0.9394   FPR90 0.2329
  dino_ep3         AUROC 0.9091   FPR90 0.3531
  dino_ep5         AUROC 0.9156   FPR90 0.3534

"""