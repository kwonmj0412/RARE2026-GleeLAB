#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ensemble_harness.py — 멤버별 전처리 앙상블 채점 (Step 53)
========================================================
접근 A 검증: GastroDINO 만 WB+CLAHE, 나머지(RN50/DINOv3) 는 원본(none).
전처리 다양성이 앙상블 다양성의 새 축이 되는가? 재학습 0.

각 멤버 = (name, backbone, ckpt_dir, weights, size, color_const, lora_r, weight)
앙상블 = z-정규화 로짓 가중평균 (제출 파이프라인과 동일 원리).

external 검증셋(EVC+EDD+HK) + 각 멤버 개별 + 앙상블 채점.
record(다 none) vs WB+CLAHE앙상블(GastroDINO만 wb_clahe) 비교.

멤버 스펙 파일 (JSON):
  [
    {"name":"rn50","backbone":"resnet50","ckpt":"./runs_5f_rn50_gastro5m",
     "weights":"<RN50_DINOv1>","size":768,"cc":"none","lora":0,"w":0.5},
    {"name":"gastro","backbone":"dinov3_vitl","ckpt":"./runs_5f_dinov3_gastro_wbclahe",
     "weights":"./runs_dino_dinov3_500k/dinov3_dino_ep1.pt","size":768,"cc":"wb_clahe","lora":8,"w":0.5},
    {"name":"dinov3","backbone":"dinov3_vitl","ckpt":"./runs_5f_dinov3_res768",
     "weights":"none","size":768,"cc":"none","lora":8,"w":1.5}
  ]

사용법:
  python ensemble_harness.py --members members_wbclahe.json \
    --evc-root ... --edd-manifest ... --hk-root ...
"""
import argparse, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).parent))
import train_baseline as tb
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except Exception:
    pass


def fpr90(y, s):
    fpr, tpr, _ = roc_curve(y, s); return float(np.interp(0.9, tpr, fpr))


def ppv90(f, prev=0.01):
    return 0.9 * prev / (0.9 * prev + f * (1 - prev) + 1e-12)


@torch.no_grad()
def member_logits(m, df, size_default, preset, dev):
    """한 멤버(백본+전처리)로 df 채점 → z-정규화 로짓 [N] (5-fold 평균)."""
    import dataclasses
    size = m.get("size", size_default)
    cfg = dataclasses.replace(tb.PRESETS[preset], color_const=m.get("cc", "none"))
    cks = sorted(Path(m["ckpt"]).glob("*kfold*/best.pt")) or sorted(Path(m["ckpt"]).glob("**/best.pt"))
    if not cks:
        raise FileNotFoundError(f"체크포인트 없음: {m['ckpt']}")
    ds = tb.RareDS(df, Path("/"), None, size, False, cfg, None, False)
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4)
    fold_scores, y = [], None
    for ck in cks:
        net = tb.Net(m.get("weights", "none"), cfg.pool, cfg.topk_frac, cfg.mask_aware,
                     backbone=m["backbone"], lora_r=m.get("lora", 8))
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        net.load_state_dict(sd, strict=False)
        net = net.to(dev).eval()
        ss, ys = [], []
        for xb, mb, yb, _ in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                v = net(xb.to(dev), mb.to(dev))
            ss.append(v.detach().float().cpu().numpy()); ys.append(yb.numpy())
        fold_scores.append(np.concatenate(ss)); y = np.concatenate(ys)
        del net
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    Z = np.stack([(s - s.mean()) / (s.std() + 1e-6) for s in fold_scores], 0)
    return y, Z.mean(0)


def build_sets(evc_root, edd, hk_root):
    d = {}
    if evc_root:
        imgs = sorted((Path(evc_root) / "images").glob("*.png"))
        d["EVC"] = pd.DataFrame([{"rel": str(p), "label": 1 if "ACHD" in p.stem.upper() else 0}
                                 for p in imgs if ("ACHD" in p.stem.upper() or "NDBT" in p.stem.upper())])
    if edd and Path(edd).exists():
        d["EDD"] = pd.read_csv(edd)[["rel", "label"]].copy()
    if hk_root:
        base = Path(hk_root) / "upper-gi-tract" / "pathological-findings"
        rows = []
        for sub in ("barretts", "barretts-short-segment"):
            dd = base / sub
            if dd.exists():
                rows += [{"rel": str(p), "label": 0} for p in sorted(dd.glob("*.jpg"))]
        d["HK_barr"] = pd.DataFrame(rows)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", required=True, help="멤버 스펙 JSON")
    ap.add_argument("--evc-root", default=None)
    ap.add_argument("--edd-manifest", default=None)
    ap.add_argument("--hk-root", default=None)
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--preset", default="topk")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    members = json.load(open(args.members))
    print(f"[멤버] {len(members)}개: " + ", ".join(
        f"{m['name']}({m['backbone']},cc={m.get('cc','none')},w={m.get('w',1.0)})" for m in members))

    sets = build_sets(args.evc_root, args.edd_manifest, args.hk_root)
    for sn, df in sets.items():
        print(f"[외부셋] {sn}: {len(df)}장 (양성 {int((df['label']==1).sum())})")

    # 각 셋에서 멤버별 로짓 → 개별 + 앙상블
    print("\n" + "=" * 70)
    print("멤버별 + 앙상블 (멤버별 전처리)")
    print("=" * 70)
    for sn, df in sets.items():
        print(f"\n[{sn}]")
        member_scores = {}
        y_ref = None
        for m in members:
            y, s = member_logits(m, df, args.size, args.preset, dev)
            member_scores[m["name"]] = (s, m.get("w", 1.0))
            y_ref = y
            if len(np.unique(y)) > 1:
                print(f"  {m['name']:10s}(cc={m.get('cc','none'):8s}): "
                      f"AUROC {roc_auc_score(y,s):.4f}  fpr90 {fpr90(y,s):.4f}")
            else:
                hi = float((s > np.percentile(s, 90)).mean())
                print(f"  {m['name']:10s}(cc={m.get('cc','none'):8s}): [음성전용] 고점수비율 {hi:.3f}")
        # 가중 앙상블
        ens = sum(w * s for s, w in member_scores.values()) / sum(w for _, w in member_scores.values())
        if len(np.unique(y_ref)) > 1:
            print(f"  {'앙상블':10s}          : AUROC {roc_auc_score(y_ref,ens):.4f}  "
                  f"fpr90 {fpr90(y_ref,ens):.4f}  ppv@90R(1%) {ppv90(fpr90(y_ref,ens)):.4f}")
        else:
            hi = float((ens > np.percentile(ens, 90)).mean())
            print(f"  {'앙상블':10s}          : [음성전용] 고점수비율 {hi:.3f}")

    print("\n[판정] WB+CLAHE앙상블(GastroDINO만 wb_clahe)이 EDD/통합에서 강하면")
    print("       → 멤버별 전처리(접근 A) 유효 → 제출. 원본 record 와 비교.")


if __name__ == "__main__":
    main()
    
"""
python ensemble_harness.py --members members_wbclahe.json \
  --evc-root <DATA_ROOT>/EVC_Barretts_Data \
  --edd-manifest ./edd2020_train_clean.csv \
  --hk-root <DATA_ROOT>/HyperKvasir/labeled-images
  

python ensemble_harness.py --members members_baseline.json \
  --evc-root <DATA_ROOT>/EVC_Barretts_Data \
  --edd-manifest ./edd2020_train_clean.csv \
  --hk-root <DATA_ROOT>/HyperKvasir/labeled-images
"""

"""
# 조합 1: ViT 2개만 WB+CLAHE (RN50 없이)
echo "===== 조합1 ViT-only WB+CLAHE ====="
python ensemble_harness.py --members members_vit_wbclahe.json \
  --evc-root <DATA_ROOT>/EVC_Barretts_Data \
  --edd-manifest ./edd2020_train_clean.csv \
  --hk-root <DATA_ROOT>/HyperKvasir/labeled-images

# 조합 3: 혼합 (RN50 원본 + ViT 2개 WB+CLAHE)
echo "===== 조합3 혼합 ====="
python ensemble_harness.py --members members_mixed_wbclahe.json \
  --evc-root <DATA_ROOT>/EVC_Barretts_Data \
  --edd-manifest ./edd2020_train_clean.csv \
  --hk-root <DATA_ROOT>/HyperKvasir/labeled-images

# 조합 2: 전 멤버 WB+CLAHE
echo "===== 조합2 전멤버 WB+CLAHE ====="
python ensemble_harness.py --members members_wbclahe_all.json \
  --evc-root <DATA_ROOT>/EVC_Barretts_Data \
  --edd-manifest ./edd2020_train_clean.csv \
  --hk-root <DATA_ROOT>/HyperKvasir/labeled-images
"""

"""
# 기준: seg 하이브리드 (현 최고, Test 0.0259/Val 0.0405)
echo "===== seg 하이브리드 (기준) ====="
python ensemble_harness.py --members members_seg_champ.json \
  --evc-root <DATA_ROOT>/EVC_Barretts_Data \
  --edd-manifest ./edd2020_train_clean.csv \
  --hk-root <DATA_ROOT>/HyperKvasir/labeled-images

# seg + CAFormer (두 축 다양성)
echo "===== seg + CAFormer ====="
python ensemble_harness.py --members members_seg_caformer.json \
  --evc-root <DATA_ROOT>/EVC_Barretts_Data \
  --edd-manifest ./edd2020_train_clean.csv \
  --hk-root <DATA_ROOT>/HyperKvasir/labeled-images
"""