#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
external_harness.py — 다중 외부 센터 검증 하네스 (Step 47)
========================================================
LOCO(우리 두 센터, 색 유사 = 약한 신호)를 넘어, 학습에 안 쓴 여러 외부
데이터셋으로 미지-센터 일반화를 측정한다. 3번의 제출 실패(로컬 OOF 함정)를
근본 해결: "모든 외부 셋에서 baseline 을 이기는 설정" = 진짜 강건.

외부 검증셋 (전부 학습에 미사용):
  EVC          : ACHD 50 양성 + NDBT 50 음성 (파일명 라벨)
  EDD2020      : 94 양성 + 66 음성 (edd2020_train_clean.csv)
  HK_barretts  : 94 음성 (barretts=NDBE) — hyperkvasir barretts/short-segment

채점: 챌린지 지표 = 1% 유병률 PPV@90R (관측 유병률 아님).
  각 셋 개별 + 통합. 여러 모델(baseline / WB+CLAHE / EDD 등) 나란히.

사용법:
  python external_harness.py \
    --models baseline:./runs_5f_dinov3_gastro:none \
             wbclahe:./runs_5f_dinov3_gastro_wbclahe:wb_clahe \
    --evc-root <DATA_ROOT>/EVC_Barretts_Data \
    --edd-manifest ./edd2020_train_clean.csv \
    --hk-root <DATA_ROOT>/HyperKvasir/labeled-images \
    --weights ./runs_dino_dinov3_500k/dinov3_dino_ep1.pt --size 768
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import train_baseline as tb


def fpr_at_90(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(0.9, tpr, fpr))


def ppv90_challenge(fpr90, prev=0.01):
    """챌린지 지표: 1% 유병률 고정 PPV@90R."""
    return 0.9 * prev / (0.9 * prev + fpr90 * (1 - prev) + 1e-12)


def build_evc_df(evc_root):
    root = Path(evc_root)
    imgs = sorted((root / "images").glob("*.png"))
    rows = []
    for p in imgs:
        st = p.stem.upper()
        lab = 1 if "ACHD" in st else (0 if "NDBT" in st else None)
        if lab is not None:
            rows.append({"rel": str(p), "label": lab})
    return pd.DataFrame(rows)


def build_hk_barretts_df(hk_root):
    root = Path(hk_root)
    base = root / "upper-gi-tract" / "pathological-findings"
    rows = []
    for sub in ("barretts", "barretts-short-segment"):
        d = base / sub
        if d.exists():
            for p in sorted(d.glob("*.jpg")):
                rows.append({"rel": str(p), "label": 0})  # barretts = NDBE
    return pd.DataFrame(rows)


@torch.no_grad()
def _load_oof_logits(oof_dir):
    """OOF preds.csv 에서 (filename->logit, filename->label) 로드. t,b 학습용."""
    import glob as _g
    cands = _g.glob(str(Path(oof_dir) / "**" / "preds.csv"), recursive=True)
    if not cands:
        return None, None
    clf = pd.read_csv(cands[0])
    lcol = next((c for c in ("logit", "score", "pred", "prob") if c in clf.columns), clf.columns[-1])
    ycol = next((c for c in ("label", "y", "target") if c in clf.columns), None)
    logit = clf[lcol].to_numpy(float)
    if ycol:
        y = clf[ycol].to_numpy(int)
    else:
        y = np.array([1 if "neo" in str(v).lower() else 0
                      for v in clf.get("filename", clf.iloc[:, 0])])
    return logit, y


@torch.no_grad()
def score_df(df, ckpt_dir, color_const, weights, size, backbone, lora_r, preset, dev,
             agg="mean_z", oof_dir=None):
    """5-fold 체크포인트로 df 채점. color_const 적용. agg 로 집계 방식 선택.
    agg in {mean_z, mean_raw, recal_mean, recal_noisyor}."""
    cfg = tb.PRESETS[preset]
    import dataclasses
    cfg = dataclasses.replace(cfg, color_const=color_const)

    cks = sorted(Path(ckpt_dir).glob("*kfold*/best.pt")) or sorted(Path(ckpt_dir).glob("**/best.pt"))
    if not cks:
        raise FileNotFoundError(f"체크포인트 없음: {ckpt_dir}")

    ds = tb.RareDS(df, Path("/"), None, size, False, cfg, None, False)
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=8)

    fold_scores = []
    for ck in cks:
        net = tb.Net(weights, cfg.pool, cfg.topk_frac, cfg.mask_aware,
                     backbone=backbone, lora_r=lora_r)
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        net.load_state_dict(sd, strict=False)
        net = net.to(dev).eval()
        ss, ys = [], []
        for xb, mb, yb, _ in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                v = net(xb.to(dev), mb.to(dev))
            ss.append(v.detach().float().cpu().numpy()); ys.append(yb.numpy())
        fold_scores.append(np.concatenate(ss))
        y = np.concatenate(ys)
        del net
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    # 집계 방식 선택.
    # mean_z (기본, 챔피언 사용): z-정규화 후 fold 평균. 외부 의존 없음.
    # recal_* (선택): 온도/바이어스 재보정. 랭크 기반 지표에는 영향이 없어
    #   최종 챔피언에는 미사용. recal_pool 모듈이 있을 때만 활성화된다.
    def _mean_z(fs):
        Z = np.stack([(s - s.mean()) / (s.std() + 1e-6) for s in fs], 0)
        return Z.mean(0)

    if agg in ("recal_mean", "recal_noisyor"):
        try:
            import recal_pool as rp
        except ImportError:
            print("    [경고] recal_pool 모듈 없음 -> mean_z 로 대체")
            return y, _mean_z(fold_scores), len(cks)
        ol, oy = _load_oof_logits(oof_dir) if oof_dir else (None, None)
        if ol is None:
            print(f"    [경고] OOF 없음({oof_dir}) -> mean_z 로 대체")
            return y, _mean_z(fold_scores), len(cks)
        t, b = rp.fit_temp_bias(ol, oy)
        tbs = [(t, b)] * len(fold_scores)
        sc = rp.aggregate(fold_scores, method=agg, tb_params=tbs)
    elif agg == "mean_raw":
        sc = np.mean(np.stack(fold_scores, 0), 0)
    else:  # mean_z (기본)
        sc = _mean_z(fold_scores)
    return y, sc, len(cks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="name:ckpt_dir:color_const[:agg[:oof_dir]]  "
                         "agg in {mean_z,mean_raw,recal_mean,recal_noisyor}. "
                         "예: recal:./runs_...:none:recal_noisyor:./runs_..._oof")
    ap.add_argument("--evc-root", default=None)
    ap.add_argument("--edd-manifest", default=None)
    ap.add_argument("--hk-root", default=None)
    ap.add_argument("--weights", default="none")
    ap.add_argument("--backbone", default="dinov3_vitl")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--preset", default="topk")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 외부 셋 구성
    sets = {}
    if args.evc_root:
        sets["EVC"] = build_evc_df(args.evc_root)
    if args.edd_manifest:
        d = pd.read_csv(args.edd_manifest)
        sets["EDD"] = d[["rel", "label"]].copy()
    if args.hk_root:
        sets["HK_barr"] = build_hk_barretts_df(args.hk_root)
    for name, df in sets.items():
        print(f"[외부셋] {name}: {len(df)}장 (양성 {int((df['label']==1).sum())}, "
              f"음성 {int((df['label']==0).sum())})")

    # 모델별 채점
    results = {}  # model -> {set -> (fpr90, auroc, ppv)}
    preds = {}    # model -> {set -> (y, s)}
    for spec in args.models:
        parts = spec.split(":")
        name, ckpt_dir, cc = parts[0], parts[1], parts[2]
        agg = parts[3] if len(parts) > 3 else "mean_z"
        oof_dir = parts[4] if len(parts) > 4 else ckpt_dir
        print(f"\n===== 모델 [{name}] (color_const={cc}, agg={agg}) =====")
        results[name] = {}; preds[name] = {}
        for sname, df in sets.items():
            y, s, nck = score_df(df, ckpt_dir, cc, args.weights, args.size,
                                 args.backbone, args.lora_r, args.preset, dev,
                                 agg=agg, oof_dir=oof_dir)
            preds[name][sname] = (y, s)
            if len(np.unique(y)) < 2:
                # 음성전용 셋(HK_barr): fpr90/auroc 정의 안 됨.
                # 대신 위양성 경향 = 점수 상위 10% 비율 (통합 pool 에서 위양성으로 기여)
                hi = float((s > np.percentile(s, 90)).mean())
                results[name][sname] = (np.nan, np.nan, np.nan)
                print(f"  {sname:8s}: [음성전용 {len(y)}장] 고점수(>90%ile) 비율 {hi:.3f} "
                      f"— 통합에서 위양성 기여 ({nck} ckpts)")
            else:
                f = fpr_at_90(y, s); auc = roc_auc_score(y, s); ppv = ppv90_challenge(f)
                results[name][sname] = (f, auc, ppv)
                print(f"  {sname:8s}: fpr90 {f:.4f}  auroc {auc:.4f}  ppv@90R(1%) {ppv:.4f}  ({nck} ckpts)")

    # 통합 채점 (모든 외부 셋 pool, z-정규화 후)
    print("\n" + "=" * 70)
    print("통합 (모든 외부 셋 pooled, z-정규화)")
    print("=" * 70)
    for name in results:
        ys, ss = [], []
        for sname, (y, s) in preds[name].items():
            z = (s - s.mean()) / (s.std() + 1e-6)
            ys.append(y); ss.append(z)
        Y = np.concatenate(ys); S = np.concatenate(ss)
        f = fpr_at_90(Y, S); auc = roc_auc_score(Y, S); ppv = ppv90_challenge(f)
        print(f"  {name:10s}: fpr90 {f:.4f}  auroc {auc:.4f}  ppv@90R(1%) {ppv:.4f}")

    # 비교표
    print("\n" + "=" * 70)
    print("★ 모델 비교 (fpr90, 낮을수록 좋음) — 모든 셋에서 이기면 진짜 강건")
    print("=" * 70)
    snames = list(sets.keys())
    print(f"  {'model':12s} " + " ".join(f"{s:>9s}" for s in snames) + f" {'통합':>9s}")
    for name in results:
        row = [results[name][s][0] for s in snames]
        # 통합
        ys, ss = [], []
        for sname, (y, s) in preds[name].items():
            z = (s - s.mean()) / (s.std() + 1e-6); ys.append(y); ss.append(z)
        allf = fpr_at_90(np.concatenate(ys), np.concatenate(ss))
        cells = " ".join(("     nan " if np.isnan(v) else f"{v:9.4f}") for v in row)
        print(f"  {name:12s} " + cells + f" {allf:9.4f}")

    print("\n[판정] baseline 대비 모든(또는 대부분) 외부 셋에서 fpr90 낮으면")
    print("       → 진짜 미지-센터 개선 → 제출 확신. LOCO 하나보다 강한 신호.")


if __name__ == "__main__":
    main()