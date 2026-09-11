#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mae_linear_probe.py — 2단계 게이트: MAE 표현 개선 측정 (Step 31)
=============================================================
"MAE continued pretraining 이 표현을 개선했는가?"를 linear probe 로 측정.
  백본 동결 -> RARE26 특징 추출 -> 5-fold 선형분류 -> OOF AUROC.
  원본 DINOv3 vs MAE 각 체크포인트 비교.

★ fold 노이즈 주의 (우리가 확립한 사실):
  fold별 AUROC 는 SD ~0.02 로 흔들림. 단일 숫자로 판단 금지.
  - 5-fold 평균 + 원본 대비 '방향'만 본다.
  - 여러 MAE 체크포인트(1,2,3,5ep)에서 단조 개선/악화 패턴을 본다.
    (오르다 내려가면 MAE 가 표현을 파괴하기 시작한 지점)

게이트 판정:
  - MAE 최고점 AUROC > 원본 + 0.01 (여러 ep 에서 일관) -> 신호 있음, 3단계 진행
  - 원본과 비슷하거나 하락 -> MAE 무효 또는 표현 파괴 -> 접거나 DINO/iBOT 재고

사용법:
  # 원본 기준선
  python mae_linear_probe.py --backbone-ckpt none \
    --data-root <DATA_ROOT>/clean --folds ./folds_step3/folds.csv \
    --out ./mae_probe --tag original
  # MAE 각 체크포인트
  python mae_linear_probe.py --backbone-ckpt ./runs_mae_dinov3/dinov3_mae_ep3.pt \
    --data-root ... --folds ... --out ./mae_probe --tag mae_ep3

의존성: torch, timm, opencv, numpy, pandas, scikit-learn
"""
from __future__ import annotations
import argparse, glob
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

cv2.setNumThreads(0)
IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)


@torch.no_grad()
def extract_features(enc, paths, size, dev, n_prefix=5, bs=32):
    """CLS 토큰(전역 표현)을 특징으로. linear probe 용."""
    feats = []
    for i in range(0, len(paths), bs):
        batch = []
        for p in paths[i:i+bs]:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
            x = (rgb.astype(np.float32)/255.0 - IMNET_MEAN)/IMNET_STD
            batch.append(x.transpose(2, 0, 1))
        xb = torch.from_numpy(np.stack(batch)).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type=="cuda"):
            f = enc.forward_features(xb)      # (B, n_prefix+N, C)
            cls = f[:, 0, :]                  # CLS 토큰
        feats.append(cls.float().cpu().numpy())
    return np.concatenate(feats)


def fpr90(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(0.9, tpr, fpr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone-ckpt", required=True, help="MAE .pt 또는 'none'(원본)")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default="./mae_probe")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    import timm
    enc = timm.create_model("vit_large_patch16_dinov3.lvd1689m",
                            pretrained=True, num_classes=0, img_size=args.size).to(dev).eval()
    if args.backbone_ckpt.lower() != "none":
        sd = torch.load(args.backbone_ckpt, map_location=dev, weights_only=False)
        missing, unexpected = enc.load_state_dict(sd, strict=False)
        print(f"[ckpt] {Path(args.backbone_ckpt).name} 로드 "
              f"(missing {len(missing)}, unexpected {len(unexpected)})")
    else:
        print("[ckpt] 원본 DINOv3 (기준선)")

    # 데이터 인덱스
    df = pd.read_csv(args.folds)
    fcol = next(c for c in ("fold","kfold","fold_id") if c in df.columns)
    ncol = next(c for c in ("filename","file","name","rel","path") if c in df.columns)

    def find_path(fn):
        for cen in ("center_1","center_2"):
            for cls in ("ndbe","neo"):
                p = Path(args.data_root)/cen/cls/fn
                if p.exists():
                    return p, (1 if cls=="neo" else 0)
        hits = glob.glob(str(Path(args.data_root)/"**"/fn), recursive=True)
        if hits:
            return Path(hits[0]), (1 if "neo" in hits[0] else 0)
        return None, None

    paths, labels, folds = [], [], []
    for _, r in df.iterrows():
        fn = Path(str(r[ncol])).name
        p, lab = find_path(fn)
        if p is not None:
            paths.append(p); labels.append(lab); folds.append(int(r[fcol]))
    y = np.array(labels); fold = np.array(folds)
    print(f"[data] {len(paths)}장 (양성 {int(y.sum())})")

    print("[extract] 특징 추출 중...")
    X = extract_features(enc, paths, args.size, dev)
    print(f"[extract] 특징 {X.shape}")

    # 5-fold linear probe (OOF)
    oof = np.full(len(y), np.nan)
    fold_aucs = []
    for k in sorted(set(fold.tolist())):
        te = fold == k; tr = ~te
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        # 표준화
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        clf.fit((X[tr]-mu)/sd, y[tr])
        oof[te] = clf.predict_proba((X[te]-mu)/sd)[:, 1]
        fa = roc_auc_score(y[te], oof[te])
        fold_aucs.append(fa)
    auc = roc_auc_score(y, oof)
    f90 = fpr90(y, oof)
    print(f"\n[{args.tag}] linear probe:")
    print(f"  OOF AUROC = {auc:.4f}")
    print(f"  fold별 AUROC = {[f'{a:.3f}' for a in fold_aucs]} (SD {np.std(fold_aucs):.3f})")
    print(f"  OOF FPR90 = {f90:.4f}")

    # 결과 누적 저장
    res = out / "probe_results.csv"
    row = pd.DataFrame([{"tag": args.tag, "auroc": auc, "fpr90": f90,
                         "fold_sd": np.std(fold_aucs),
                         "ckpt": args.backbone_ckpt}])
    if res.exists():
        row = pd.concat([pd.read_csv(res), row], ignore_index=True)
    row.to_csv(res, index=False)
    print(f"\n[saved] {res}")
    if len(row) > 1:
        print("\n[누적 비교]")
        base = row[row.tag == "original"]
        b = base.auroc.iloc[0] if len(base) else None
        for r in row.itertuples():
            d = f"({r.auroc-b:+.4f})" if b is not None and r.tag != "original" else ""
            print(f"  {r.tag:16s} AUROC {r.auroc:.4f} {d}  FPR90 {r.fpr90:.4f}")
        if b is not None:
            best = row[row.tag != "original"].auroc.max() if (row.tag != "original").any() else b
            print(f"\n  [게이트] 원본 {b:.4f} vs MAE 최고 {best:.4f} "
                  f"({best-b:+.4f})")
            if best > b + 0.01:
                print("  ✅ 표현 개선 신호. 3단계(본학습) 진행 검토.")
            elif best < b - 0.005:
                print("  ⚠️ 표현 악화. MAE 가 DINOv3 표현을 파괴 -> 접거나 lr/freeze 재조정.")
            else:
                print("  ~ 개선 미미. fold 노이즈 범위. 신중히 판단.")


if __name__ == "__main__":
    main()
    
"""
python mae_linear_probe.py --backbone-ckpt none \
  --data-root <DATA_ROOT>/clean --folds ./folds_step3/folds.csv \
  --size 224 --out ./mae_probe --tag original
  
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
[ckpt] 원본 DINOv3 (기준선)
[data] 3095장 (양성 158)
[extract] 특징 추출 중...
[extract] 특징 (3095, 1024)

[original] linear probe:
  OOF AUROC = 0.9307
  fold별 AUROC = ['0.937', '0.968', '0.891', '0.947', '0.917'] (SD 0.026)
  OOF FPR90 = 0.2057

[saved] mae_probe/probe_results.csv

"""
"""
for EP in 1 2 3 5; do
  python mae_linear_probe.py --backbone-ckpt ./runs_mae_dinov3/dinov3_mae_ep${EP}.pt \
    --data-root <DATA_ROOT>/clean --folds ./folds_step3/folds.csv \
    --size 224 --out ./mae_probe --tag mae_ep${EP}
done

[ckpt] dinov3_mae_ep1.pt 로드 (missing 0, unexpected 0)
[data] 3095장 (양성 158)
[extract] 특징 추출 중...
[extract] 특징 (3095, 1024)

[mae_ep1] linear probe:
  OOF AUROC = 0.6933
  fold별 AUROC = ['0.612', '0.734', '0.706', '0.712', '0.704'] (SD 0.042)
  OOF FPR90 = 0.8100

[saved] mae_probe/probe_results.csv

[누적 비교]
  original         AUROC 0.9307   FPR90 0.2057
  mae_ep1          AUROC 0.6933 (-0.2375)  FPR90 0.8100

  [게이트] 원본 0.9307 vs MAE 최고 0.6933 (-0.2375)
  ⚠️ 표현 악화. MAE 가 DINOv3 표현을 파괴 -> 접거나 lr/freeze 재조정.
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
[ckpt] dinov3_mae_ep2.pt 로드 (missing 0, unexpected 0)
[data] 3095장 (양성 158)
[extract] 특징 추출 중...
[extract] 특징 (3095, 1024)

[mae_ep2] linear probe:
  OOF AUROC = 0.6776
  fold별 AUROC = ['0.656', '0.694', '0.677', '0.756', '0.608'] (SD 0.048)
  OOF FPR90 = 0.8008

[saved] mae_probe/probe_results.csv

[누적 비교]
  original         AUROC 0.9307   FPR90 0.2057
  mae_ep1          AUROC 0.6933 (-0.2375)  FPR90 0.8100
  mae_ep2          AUROC 0.6776 (-0.2531)  FPR90 0.8008

  [게이트] 원본 0.9307 vs MAE 최고 0.6933 (-0.2375)
  ⚠️ 표현 악화. MAE 가 DINOv3 표현을 파괴 -> 접거나 lr/freeze 재조정.
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
[ckpt] dinov3_mae_ep3.pt 로드 (missing 0, unexpected 0)
[data] 3095장 (양성 158)
[extract] 특징 추출 중...
[extract] 특징 (3095, 1024)

[mae_ep3] linear probe:
  OOF AUROC = 0.6917
  fold별 AUROC = ['0.661', '0.690', '0.702', '0.757', '0.652'] (SD 0.037)
  OOF FPR90 = 0.8284

[saved] mae_probe/probe_results.csv

[누적 비교]
  original         AUROC 0.9307   FPR90 0.2057
  mae_ep1          AUROC 0.6933 (-0.2375)  FPR90 0.8100
  mae_ep2          AUROC 0.6776 (-0.2531)  FPR90 0.8008
  mae_ep3          AUROC 0.6917 (-0.2390)  FPR90 0.8284

  [게이트] 원본 0.9307 vs MAE 최고 0.6933 (-0.2375)
  ⚠️ 표현 악화. MAE 가 DINOv3 표현을 파괴 -> 접거나 lr/freeze 재조정.
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
[ckpt] dinov3_mae_ep5.pt 로드 (missing 0, unexpected 0)
[data] 3095장 (양성 158)
[extract] 특징 추출 중...
[extract] 특징 (3095, 1024)

[mae_ep5] linear probe:
  OOF AUROC = 0.7052
  fold별 AUROC = ['0.649', '0.708', '0.724', '0.770', '0.671'] (SD 0.042)
  OOF FPR90 = 0.7981

[saved] mae_probe/probe_results.csv

[누적 비교]
  original         AUROC 0.9307   FPR90 0.2057
  mae_ep1          AUROC 0.6933 (-0.2375)  FPR90 0.8100
  mae_ep2          AUROC 0.6776 (-0.2531)  FPR90 0.8008
  mae_ep3          AUROC 0.6917 (-0.2390)  FPR90 0.8284
  mae_ep5          AUROC 0.7052 (-0.2255)  FPR90 0.7981

  [게이트] 원본 0.9307 vs MAE 최고 0.7052 (-0.2255)
  ⚠️ 표현 악화. MAE 가 DINOv3 표현을 파괴 -> 접거나 lr/freeze 재조정.

"""