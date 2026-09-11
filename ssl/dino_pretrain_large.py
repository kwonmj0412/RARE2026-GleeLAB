#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dino_pretrain_large.py — GastroDINO 대규모 재학습 (Step 45)
=====================================================
검증된 dino_pretrain.py 를 대규모(260만장)용으로 확장.
목표: GastroDINO(50만×1ep, probe 0.99)를 260만 전체로 더 강하게.
      최강 멤버를 강화 -> 앙상블 전체 상승 (조합 최적화 한계 돌파).

기존 대비 변경 (대규모 안정성):
  1. n_images 260만 (전체) — zip 당 더 많이 샘플
  2. warmup + cosine lr 스케줄 (고정 lr -> 안정적 수렴)
  3. teacher momentum 0.996 -> 0.9995 (긴 학습일수록 느린 갱신, DINO 논문)
  4. 자주 저장 (매 0.25ep = iteration 기준) -> 중간 probe 로 최적 선택
  5. bs 최대화 옵션 (Blackwell 96GB)
  6. resume 지원 (긴 학습 중단 대비)

검증된 것 보존:
  - student/teacher 둘 다 원본 DINOv3 출발
  - multi-crop (global 2 + local 6)
  - teacher 독립생성+load_state_dict (deepcopy 아님, weight_norm 호환)

사용법:
  python dino_pretrain_large.py \
    --gastro-dir <DATA_ROOT>/Gastronet-5M \
    --n-images 2600000 --epochs 2 --lr 1e-5 --warmup-frac 0.1 \
    --bs 96 --teacher-momentum 0.9995 --save-every-frac 0.25 \
    --out ./runs_dino_large --workers 16

* 시간 오래 걸림 (하루+). 중간 체크포인트를 probe 로 확인하며 진행.
* n-images 를 처음엔 작게(50만) 테스트 후 전체로 권장.
"""
from __future__ import annotations
import argparse, glob, math, random, time, zipfile
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 검증된 컴포넌트 재사용
from dino_components import DINOHead, DINOLoss, ema_update
from dino_pretrain import (rand_resized_crop, color_jitter, MultiCropDataset,
                           collate_multicrop, DINOWrap)


def build_index_large(gastro_dir, n_images, seed=0):
    """대규모 인덱싱: zip 당 균등 샘플. n_images 크면 zip 전체도 활용."""
    zips = sorted(glob.glob(str(Path(gastro_dir) / "*.zip")))
    zips = [z for z in zips if Path(z).stat().st_size > 1000]
    if not zips:
        # zip 아니라 png 직접일 수도
        pngs = sorted(glob.glob(str(Path(gastro_dir) / "**" / "*.png"), recursive=True))
        rng = random.Random(seed)
        rng.shuffle(pngs)
        return [("__direct__", p) for p in pngs[:n_images]]
    rng = random.Random(seed)
    per_zip = max(1, n_images // len(zips))
    index = []
    print(f"[index] {len(zips)}개 zip, zip당 최대 {per_zip:,}장 목표")
    for zp in zips:
        with zipfile.ZipFile(zp) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".png")]
            take = min(per_zip, len(names))
            index.extend((zp, nm) for nm in rng.sample(names, take))
    rng.shuffle(index)
    return index[:n_images]


def cosine_warmup_lr(step, total_steps, base_lr, warmup_steps, min_lr_frac=0.01):
    """warmup 후 cosine decay. 대규모 학습 안정성."""
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * (min_lr_frac + (1 - min_lr_frac) * cos)


def momentum_schedule(step, total_steps, base_m, final_m=0.99999):
    """teacher momentum cosine 증가 (DINO: 0.996 -> 1.0 방향)."""
    progress = step / max(1, total_steps)
    return final_m - (final_m - base_m) * (0.5 * (1 + math.cos(math.pi * progress)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gastro-dir", required=True)
    ap.add_argument("--n-images", type=int, default=2600000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-6)          # 원본 5e-6 (1e-5 아님!)
    ap.add_argument("--warmup-frac", type=float, default=0.0)  # 원본은 warmup 없음 (기본 끔)
    ap.add_argument("--cosine", action="store_true")           # cosine 옵션 (기본 끔, 고정 lr)
    ap.add_argument("--bs", type=int, default=64)              # 원본 64
    ap.add_argument("--n-local", type=int, default=6)
    ap.add_argument("--out-dim", type=int, default=8192)
    ap.add_argument("--teacher-momentum", type=float, default=0.996)  # 원본 0.996 (0.9995 아님!)
    ap.add_argument("--momentum-schedule", action="store_true")  # momentum 증가 옵션 (기본 끔)
    ap.add_argument("--teacher-temp", type=float, default=0.04)
    ap.add_argument("--freeze-blocks", type=int, default=0)
    ap.add_argument("--save-every-frac", type=float, default=0.25,
                    help="에폭의 몇 분율마다 저장 (0.25=4번/ep)")
    ap.add_argument("--out", default="./runs_dino_large")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--resume", default="", help="student 체크포인트에서 재개")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    import timm
    print(f"[index] {args.n_images:,}장 인덱싱 (대규모)...")
    index = build_index_large(args.gastro_dir, args.n_images)
    print(f"[index] {len(index):,}장 확보")

    ds = MultiCropDataset(index, n_local=args.n_local)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True, collate_fn=collate_multicrop,
                    persistent_workers=True, prefetch_factor=4)

    steps_per_ep = len(ds) // args.bs
    total_steps = steps_per_ep * args.epochs
    warmup_steps = int(total_steps * args.warmup_frac)
    save_every = max(1, int(steps_per_ep * args.save_every_frac))
    print(f"[schedule] {steps_per_ep:,} step/ep x {args.epochs}ep = {total_steps:,} step")
    print(f"[schedule] warmup {warmup_steps:,} step, 저장 매 {save_every:,} step")

    # student: 원본 DINOv3 출발 (검증된 방식)
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

    # teacher: 독립 생성 후 가중치 복사 (deepcopy 아님 - weight_norm 호환)
    teacher_bb = timm.create_model("vit_large_patch16_dinov3.lvd1689m",
                                   pretrained=True, num_classes=0, img_size=224).to(dev)
    teacher_head = DINOHead(embed_dim, args.out_dim).to(dev)
    teacher = DINOWrap(teacher_bb, teacher_head).to(dev)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    loss_fn = DINOLoss(args.out_dim, teacher_temp=args.teacher_temp).to(dev)
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.04)

    start_step = 0
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        student.backbone.load_state_dict(ck.get("student_bb", ck), strict=False)
        if "teacher_bb" in ck:
            teacher.backbone.load_state_dict(ck["teacher_bb"], strict=False)
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start_step = ck.get("step", 0)
        print(f"[resume] step {start_step:,} 에서 재개")

    print(f"[model] 학습 파라미터 {sum(p.numel() for p in params)/1e6:.1f}M, "
          f"base_lr {args.lr}, momentum {args.teacher_momentum}->0.99999")

    n_global = 2
    step = start_step
    t0 = time.time(); tot = 0.0; nb = 0
    student.train(); teacher.train()
    for ep in range(1, args.epochs + 1):
        for crops in dl:
            # 스케줄 적용
            lr = cosine_warmup_lr(step, total_steps, args.lr, warmup_steps)
            for g in opt.param_groups:
                g["lr"] = lr
            m = momentum_schedule(step, total_steps, args.teacher_momentum)

            crops = [c.to(dev, non_blocking=True) for c in crops]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type=="cuda"):
                student_out = student(crops)
                with torch.no_grad():
                    teacher_out = teacher(crops[:n_global])
                loss = loss_fn(student_out, teacher_out)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 3.0)
            opt.step()
            ema_update(student, teacher, m)
            tot += float(loss); nb += 1; step += 1

            if nb % 50 == 0:
                rate = nb * args.bs / (time.time() - t0)
                eta_h = (total_steps - step) * args.bs / max(1, rate) / 3600
                print(f"  ep{ep} step{step:,}/{total_steps:,} loss={tot/nb:.4f} "
                      f"lr={lr:.2e} m={m:.5f} {rate:.0f}img/s ETA {eta_h:.1f}h")

            # 주기적 저장 (iteration 기준 - 중간 probe 용)
            if step % save_every == 0:
                frac = step / steps_per_ep
                sd = {k: (v.half() if v.is_floating_point() else v)
                      for k, v in teacher.backbone.state_dict().items()}
                tag = f"dinov3_large_step{step}"
                torch.save(sd, out / f"{tag}.pt")
                # resume 용 풀 상태도 (최신 1개만)
                torch.save({"student_bb": student.backbone.state_dict(),
                            "teacher_bb": teacher.backbone.state_dict(),
                            "opt": opt.state_dict(), "step": step},
                           out / "resume_latest.pt")
                print(f"  [저장] {tag}.pt (frac {frac:.2f}ep, teacher backbone)")

        print(f"[ep{ep}] loss={tot/nb:.4f} ({(time.time()-t0)/3600:.1f}h 누적)")

    # 최종 저장
    sd = {k: (v.half() if v.is_floating_point() else v)
          for k, v in teacher.backbone.state_dict().items()}
    torch.save(sd, out / "dinov3_large_final.pt")
    print(f"\n[완료] {out}/")
    print("  중간/최종 체크포인트를 probe 로 확인:")
    print("  for CK in $(ls " + str(out) + "/dinov3_large_*.pt); do")
    print("    python mae_linear_probe.py --backbone-ckpt $CK \\")
    print("      --data-root ... --folds ... --size 224 --out ./large_probe --tag $(basename $CK)")
    print("  done")
    print("  * 원본 GastroDINO probe 와 비교. 넘으면 앙상블 교체.")


if __name__ == "__main__":
    main()
    
"""
cd <DATA_ROOT>

# SSL 체크포인트 4개 + 최종 probe
for CK in ./runs_dino_continue/dinov3_large_step1952.pt \
          ./runs_dino_continue/dinov3_large_step3904.pt \
          ./runs_dino_continue/dinov3_large_step5856.pt \
          ./runs_dino_continue/dinov3_large_step7808.pt \
          ./runs_dino_continue/dinov3_large_final.pt; do
  echo "===== probe $(basename $CK) ====="
  python mae_linear_probe.py --backbone-ckpt $CK \
    --data-root <DATA_ROOT>/clean --folds ./folds_step3/folds.csv \
    --size 224 --out ./large_probe --tag $(basename $CK .pt)
done

# 기존 GastroDINO (비교 기준)
echo "===== probe 기존 GastroDINO ====="
python mae_linear_probe.py --backbone-ckpt ./runs_dino_dinov3_500k/dinov3_dino_ep1.pt \
  --data-root <DATA_ROOT>/clean --folds ./folds_step3/folds.csv \
  --size 224 --out ./large_probe --tag original_gastrodino
"""