#!/usr/bin/env bash
# =============================================================================
# build_hybrid_seg_submission.sh — seg 하이브리드 (Step 63)
#   pAUC 하이브리드에서 GastroDINO wb_pauc -> seg 버전 교체. 앙상블 EDD 0.6364->0.6212 유지 확인됨.
#   Test강점(record) + Val강점+지표정렬(wb_pauc 3개) 헤지. RN50 TTA 4->1 시간확보.
#   RN50(wb) + GastroDINO(wb) + DINOv3(wb) + gate, color_const=wb_clahe
#   전 멤버 동일 전처리 -> config 통일로 충분 (멤버별 전처리 불필요)
#   사용법: bash build_edd_submission.sh [GASTRO_SRC]
#     기본(인자없음): ./runs_5f_dinov3_gastro_edd  (EDD 버전)
#     baseline A/B:   bash build_edd_submission.sh ./runs_5f_dinov3_gastro
#   구성: RN50-GastroNet + GastroDINO(±EDD) + DINOv3 + gate  (검증최고 triple)
# =============================================================================
# RN50 + GastroDINO + 기존DINOv3 + MaxViT
# 개선점 (기존 build_dual 대비):
#   1. fold 를 성능 아닌 '균등 분산'으로 선택 (OOF 노이즈 과적합 회피)
#      3개면 fold 0,2,4 / 2개면 0,4 (5-fold 를 고르게 대표)
#   2. MaxViT(512) 소스 지원 + config 매핑
#   3. 각 백본 소스/개수를 명시적 변수로 (인자 순서 혼란 제거)
#
# 사용법 (변수를 스크립트 상단에서 편집):
#   bash build_quad_submission.sh
# =============================================================================
set -euo pipefail

# ---------- 설정 (여기만 편집) ----------
SUBMISSION="$HOME/Desktop/RARE25-Submission"

RN50_SRC="./runs_5f_rn50_gastro5m";     N_RN50=2;  RN50_TTA=1;  W_RN50=0.5   # 원본색, TTA1 시간확보
GASTRO_SRC="./runs_5f_dinov3_gastro";   N_GASTRO=2   # 원본색 (record)
DV3_SRC="./runs_5f_dinov3_res768";  N_DV3=2   # 원본색 (record)
# WB+CLAHE+pAUC 3백본 (우리 지표 직접 최적화, Val 강점, 별도 가중치)
RN50WBP_SRC="./runs_5f_rn50_gastro5m_wbclahe_pauc";    N_RN50WBP=2;  W_RN50WBP=0.5
GWBP_SRC="./runs_5f_dinov3_gastro_wbclahe_pauc_seg";   N_GWBP=2;     W_GWBP=1.0   # EVC seg 보조 (앙상블 EDD 유지)
DV3WBP_SRC="./runs_5f_dinov3_res768_wbclahe_pauc";     N_DV3WBP=2;   W_DV3WBP=1.0
VITS_SRC="./runs_5f_vits_g5m";          N_VITS=0;  VITS_W=0.5   # 검증최고=triple, VITS 제외

# LOCO 최적 (worst-case 미지센터): G0.5 D1.5 R0.5 V0.5, 개수 통일(3)
DV3_SIZE=768        # GastroDINO + DINOv3 공통 크기
DV3_W=0.5           # GastroDINO(dinov3_) 가중치 = G0.5
DV3B_W=1.5          # DINOv3 원본(dinov3b_) 가중치 = D1.5 (가장 안정적 멤버 강조)
VITS_SIZE=768       # VITS ViT-S (dynamic_img_size 로 768 지원)
FUSION=logit
GATE=on             # on | off  (게이트: 미지센터 위양성 억제)
# -----------------------------------------

# fold 균등 분산: N 개를 5-fold 에서 고르게 뽑기
spread_folds() {
  local n=$1 total=5
  if [ "$n" -le 0 ]; then return; fi
  if [ "$n" -ge "$total" ]; then seq 0 $((total-1)); return; fi
  if [ "$n" -eq 1 ]; then echo 0; return; fi
  python3 -c "n=$n;t=$total;print(' '.join(str(round(i*(t-1)/(n-1))) for i in range(n)))"
}

copy_ckpts() {  # src, n, prefix
  local src=$1 n=$2 prefix=$3
  local folds=$(spread_folds "$n")
  echo "  --- $prefix: $src (folds: $folds) ---"
  local idx=0
  for K in $folds; do
    local S=$(ls "$src"/*_kfold${K}/best.pt 2>/dev/null | head -1)
    [ -z "$S" ] && { echo "!! $src fold${K} 없음"; exit 1; }
    python - "$S" "$SUBMISSION/model/${prefix}_fold${idx}.pt" <<'PYEOF'
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
for k in ("state_dict","model"):
    if isinstance(sd,dict) and k in sd and isinstance(sd[k],dict): sd=sd[k]
torch.save({k:(v.half() if torch.is_floating_point(v) else v) for k,v in sd.items()}, sys.argv[2])
PYEOF
    echo "    ${prefix}_fold${idx}.pt (원본 fold${K}, fp16)"
    idx=$((idx+1))
  done
}

echo ">> 4백본: RN50×$N_RN50 + GastroDINO×$N_GASTRO + DINOv3×$N_DV3 + VITS×$N_VITS (gate=$GATE)"
rm -f "$SUBMISSION"/model/*.pt
echo "기존 .pt 정리 완료"

# 백본별 복사 (파일명 접두사로 추론시 백본 감지)
#   rn50_*      -> resnet50 (768)
#   dinov3_*    -> GastroDINO (768)  [정렬상 먼저 = 게이트 max_models 최강]
#   dinov3b_*   -> 기존 DINOv3 (768)
#   vits_*      -> VITS ViT-S (768)
copy_ckpts "$RN50_SRC"   "$N_RN50"   "rn50"
copy_ckpts "$GASTRO_SRC" "$N_GASTRO" "dinov3"
copy_ckpts "$DV3_SRC"    "$N_DV3"    "dinov3b"
copy_ckpts "$RN50WBP_SRC" "$N_RN50WBP" "rn50wbclahe"      # RN50 wb+pauc
copy_ckpts "$GWBP_SRC"    "$N_GWBP"    "dinov3wbclahe"      # GastroDINO wb+pauc
copy_ckpts "$DV3WBP_SRC"  "$N_DV3WBP"  "dinov3bwbclahe"     # DINOv3 wb+pauc
copy_ckpts "$VITS_SRC"   "$N_VITS"   "vits"

# config 생성
BSI=4
cat > "$SUBMISSION/resources/config.json" <<JSON
{
  "pool": "topk",
  "topk_frac": 0.02,
  "mask_aware": false,
  "color_const": "none",
  "fov_crop": true,
  "size": 512,
  "tta": false,
  "batch_size": $BSI,
  "dual": {
    "rn50_size": 768,
    "rn50_weight": $W_RN50,
    "rn50_tta": $RN50_TTA,
    "dinov3_size": $DV3_SIZE,
    "dinov3_weight": $DV3_W,
    "dinov3_tta": 1,
    "dinov3b_weight": $DV3B_W,
    "gwb_size": $DV3_SIZE,
    "gwb_weight": $W_GWBP,
    "gwb_tta": 1,
    "rn50wbp_size": 768,
    "rn50wbp_weight": $W_RN50WBP,
    "rn50wbp_tta": 1,
    "dv3wbp_size": $DV3_SIZE,
    "dv3wbp_weight": $W_DV3WBP,
    "dv3wbp_tta": 1,
    "vits_size": $VITS_SIZE,
    "vits_weight": $VITS_W,
    "vits_tta": 1,
    "fusion": "$FUSION"$([ "$GATE" = "on" ] && echo ',
    "gate": {"size": '$DV3_SIZE', "q": 0.6, "gate_lo": 0.4, "penalty": 1.0, "shrink": 0.1, "max_models": 1}')
  },
  "note": "seg하이브리드: record원본3 + RN50(wb_pauc) + GastroDINO(wb_pauc_SEG) + DINOv3(wb_pauc). GastroDINO만 EVC seg 보조, 12모델"
}
JSON

echo ""
echo "[config] 생성 완료:"
python3 -c "import json;print(json.dumps(json.load(open('$SUBMISSION/resources/config.json'))['dual'],indent=2,ensure_ascii=False))"
cp "$SUBMISSION/resources/config.json" "$SUBMISSION/model/config.json" 2>/dev/null || true
echo ""
echo "[모델 파일]"
ls "$SUBMISSION"/model/*.pt | sed 's|.*/|  |'
echo ""
echo ">> 다음: do_test_run (로딩 확인) -> do_save (sanity, 시간!) -> 리더보드"
