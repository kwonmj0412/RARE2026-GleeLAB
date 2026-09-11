# RARE 2026 — GleeLAB (CANDOR)

**C**ross-center **A**daptive **N**eoplasia **D**etection via c**O**lor-normalized **R**obustness

Image-level detection of early Barrett's esophagus neoplasia under multi-center domain shift.
Submission for the [RARE 2026 Grand Challenge](https://rare25.grand-challenge.org/) (MICCAI 2026 EndoVis).

Team **GleeLAB**, Ajou University.

---

## Overview

The central difficulty of RARE is **domain shift**: a model trained on a couple of centers
collapses on the 12 unseen BONSAI centers, where an internal AUROC of ~0.99 can drop to
~0.65–0.79. CANDOR attacks this directly with a **diverse, color-normalized ensemble** whose
every design choice was validated on **unseen-center external data**, never on internal
cross-validation.

Key ideas:

- **Backbone diversity** — a GastroNet-SSL ResNet-50, a domain-adapted DINOv3 (GastroDINO,
  self-supervised on gastroenterology imagery), and an ImageNet DINOv3.
- **Color-normalization diversity** — each backbone is trained both on original color and on
  White-Balance + CLAHE normalized inputs, suppressing center-specific illumination/tint.
- **Lesion-aligned auxiliary** — an EVC segmentation auxiliary aligns the domain-SSL backbone
  to lesion location (helps the domain-SSL backbone specifically).
- **Partial-AUC loss** — directly optimizes the high-sensitivity region rewarded by the
  challenge metric (PPV@90%Recall).
- **Equal-weight fusion** — members are combined with equal weights to maximize diversity
  rather than fit noisy validation.
- **External selection** — every change was kept only if it improved an unseen-center harness
  (EDD2020, plus EVC / HyperKvasir) under the challenge's 1%-prevalence PPV@90%Recall.

## Results (open leaderboard)

| Split | PPV@90%Recall | AUROC | N |
|-------|:-------------:|:-----:|:--:|
| Test | 0.0274 | 0.88 | 1,573 |
| RARE25 Validation | 0.0409 | 0.93 | 1,143 |

Under 1% prevalence and multi-center shift, the model holds 90% sensitivity while suppressing
FPR to ≈0.32 — in a 1,000-patient surveillance setting it flags ~328 cases, catching 9 of 10
lesions while returning two-thirds of patients to routine follow-up.

---

## Repository structure

```
.
├── README.md
├── LICENSE                       # MIT
├── requirements.txt
├── src/
│   ├── train_baseline.py         # model (Net), datasets, losses, presets, training
│   ├── rare_model.py             # Ensemble + multiscale/TTA inference
│   ├── inference.py              # Grand Challenge entrypoint (per-prefix backbone routing)
│   ├── external_harness.py       # single-model unseen-center scoring (EDD/EVC/HK)
│   └── ensemble_harness.py       # ensemble unseen-center scoring
├── ssl/
│   └── dino_pretrain_large.py    # GastroDINO self-supervised pretraining (DINOv3-DINO)
├── configs/
│   ├── members_champ_equal.json  # champion ensemble members (equal weights)
│   └── config_seg_equal.json     # submission config.json (equal weights, gate on)
├── submission/
│   ├── Dockerfile
│   ├── do_build.sh
│   ├── do_save.sh
│   ├── do_test_run.sh
│   └── build_hybrid_seg_submission.sh   # copies fold ckpts + writes config.json
└── docs/
    └── method_report.pdf         # challenge method report
```

> **Note on weights and data.** Trained checkpoints (`best.pt`) and the challenge data are
> **not** included here (size / licensing). See *Data* and *Reproducing* below for how to
> obtain them and where to place them. Large SSL checkpoints can be attached as GitHub
> Releases if needed.

---

## Requirements

```bash
pip install -r requirements.txt
```

Developed with PyTorch (CUDA, bf16) on an NVIDIA RTX PRO 6000 (Blackwell). Inference for the
challenge runs in a Docker container on an NVIDIA T4 (see `submission/`).

## Data

RARE / BONSAI training data is obtained through the
[challenge page](https://rare25.grand-challenge.org/). External validation sets used **only**
for model selection (not training):

- **EDD2020** — endoscopy disease detection (unseen-center harness).
- **EVC** — Barrett's data with 5-expert lesion masks (segmentation auxiliary + saliency IoU).
- **HyperKvasir** — Barrett's / short-segment (negative-only false-positive check).

Self-supervised pretraining uses the **GastroNet-5M** unlabeled corpus.

Expected layout (edit paths in the scripts / configs to match your machine):

```
<DATA_ROOT>/
├── clean/                        # inpainted training images: center_{1,2}/{ndbe,neo}
├── folds_step3/folds.csv         # 5-fold split (patient-grouped, stratified)
├── EVC_Barretts_Data/            # images/ + annotations_bmp/ (5-expert masks)
├── edd2020_train_clean.csv       # EDD manifest (rel, label)
└── ...
```

---

## Reproducing

### 1. (Optional) Self-supervised pretraining — GastroDINO

```bash
python ssl/dino_pretrain_large.py \
  --gastro-dir <GASTRONET_5M_DIR> \
  --n-images 500000 --epochs 1 --lr 5e-6 --bs 64 \
  --save-every-frac 0.25 --out ./runs_dino --workers 16
```
(You may instead use the provided GastroNet-SSL weights for the ResNet-50 members.)

### 2. Train the 6 ensemble members (5-fold each)

Each member differs only in backbone / SSL weights / color-const / seg-aux. Example (the
GastroDINO + WB+CLAHE + pAUC + EVC-seg member):

```bash
python src/train_baseline.py \
  --data-root <DATA_ROOT>/clean --folds <DATA_ROOT>/folds_step3/folds.csv \
  --mask-bank <DATA_ROOT>/audit_step2/mask_bank \
  --color-const wb_clahe \
  --seg-weight 0.5 --seg-warmup 3 --evc-root <DATA_ROOT>/EVC_Barretts_Data \
  --backbone dinov3_vitl --weights <GASTRODINO_CKPT> --lora-r 8 \
  --split kfold --fold all --size 768 --preset topk_pauc \
  --epochs 20 --bs 4 --lr 3e-4 --amp bf16 --ema \
  --out-dir ./runs_5f_dinov3_gastro_wbclahe_pauc_seg
```

The six members (see `configs/members_champ_equal.json`):
`rn50`, `gastro`, `dinov3` (original color) and
`rn50_wbp`, `gastro_wbpseg`, `dinov3_wbp` (WB+CLAHE).

### 3. Validate on unseen centers (external harness)

```bash
python src/ensemble_harness.py --members configs/members_champ_equal.json \
  --evc-root <DATA_ROOT>/EVC_Barretts_Data \
  --edd-manifest <DATA_ROOT>/edd2020_train_clean.csv \
  --hk-root <DATA_ROOT>/HyperKvasir/labeled-images
```
A configuration is accepted only if it improves the external (EDD) PPV@90%Recall.

### 4. Build the submission container

```bash
cd submission
bash build_hybrid_seg_submission.sh          # copies fold 0,4 ckpts + writes config.json
cp configs/config_seg_equal.json resources/config.json   # equal-weight champion config
sudo bash do_test_run.sh                      # local T4-style run (checks the 600s budget)
bash do_save.sh                               # save container image for upload
```

The ensemble is **12 models** (6 members × folds {0,4}), logit-fusion with equal weights,
TTA off (to stay within the 600 s / 384-frame T4 budget), with an optional asymmetric
anomaly gate (`max_models=1`).

---

## Method notes / negative results

Consistent with our external-first methodology, we ran eight independent improvement attempts
beyond the final method; all were rejected or dropped after external / submission evidence
(anomaly-detection late-fusion, segmentation on ImageNet backbones, a stronger internal SSL
checkpoint, fold expansion, and fold-selection by external score). We report these as
first-class findings — the thoroughness is evidence that the submitted configuration is a
robust optimum. See `docs/method_report.pdf`.

## License

Released under the [MIT License](LICENSE).

## Acknowledgements

RARE / BONSAI challenge organizers (TUE-ARIA); GastroNet-5M; EDD2020; EVC; HyperKvasir.
