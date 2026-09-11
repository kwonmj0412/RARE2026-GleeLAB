"""
RARE26 제출용 inference.py
==========================
GC 컨테이너 엔트리포인트. /input 의 stacked TIFF/MHA 를 읽어
프레임별 neoplasia likelihood 리스트를 /output JSON 으로 쓴다.

핵심 설계:
  - 입력은 (N,512,512,3) uint8 RGB (probe 로 확인).
  - 학습(train_baseline.load_pair)과 동일 전처리: FOV crop -> 512 resize -> RGB /255 -> ImageNet norm.
  - 5-fold 가중치 logit 평균 + 기하 TTA. sigmoid 는 마지막에 한 번(순위 보존, 값 범위 [0,1]).
  - 스트리밍 배치로 메모리 폭증 방지 (25k 프레임 x 512 x 512 x 3 = 19GB 를 한 번에 안 올림).
  - GC 는 float32 정밀도 그대로 요구 -> round() 절대 금지 (동점이 PPV@90R 을 왜곡).

가중치 위치(우선순위):
  1. /opt/ml/models/  (tarball 업로드 마운트; do_test_run.sh 가 model/ 을 여기 마운트)
  2. resources/       (이미지에 구워진 경우)
  파일명 패턴: *.pt  (여러 개면 전부 앙상블)

config.json (resources/ 또는 /opt/ml/models/ 에 두면 읽음):
  {"pool":"topk","topk_frac":0.02,"mask_aware":false,"color_const":"none",
   "fov_crop":true,"size":512,"tta":true,"batch_size":16}
없으면 아래 DEFAULTS 사용 (= Step 4 승자 레시피).
"""
from pathlib import Path
import json
import sys
from glob import glob

import numpy as np
import SimpleITK

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")
MODEL_DIRS = [Path("/opt/ml/models"), Path("/opt/ml/model"),
              Path("/opt/app/resources"), RESOURCE_PATH]

DEFAULTS = dict(pool="topk", topk_frac=0.02, mask_aware=False, color_const="none",
                fov_crop=True, size=512, tta=True, batch_size=16)


# ---------------------------------------------------------------- 설정/가중치 탐색
def load_config():
    cfg = dict(DEFAULTS)
    for d in MODEL_DIRS:
        p = d / "config.json"
        if p.exists():
            try:
                cfg.update(json.loads(p.read_text()))
                print(f"[config] loaded from {p}")
                break
            except Exception as e:
                print(f"[config] {p} 파싱 실패: {e}")
    print(f"[config] {cfg}")
    return cfg


def find_checkpoints():
    seen, ckpts = set(), []
    for d in MODEL_DIRS:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.pt")) + sorted(d.glob("*.pth")):
            # 예제 resnet50.pth(133바이트 더미) 제외
            if f.stat().st_size < 10000:
                print(f"[ckpt] skip tiny {f} ({f.stat().st_size}B)")
                continue
            key = f.name
            if key not in seen:
                seen.add(key); ckpts.append(f)
    return ckpts


# ---------------------------------------------------------------- 입력 로딩
def load_stack(location: Path) -> np.ndarray:
    files = (sorted(glob(str(location / "*.tif"))) + sorted(glob(str(location / "*.tiff")))
             + sorted(glob(str(location / "*.mha"))) + sorted(glob(str(location / "*.mhd"))))
    if not files:
        raise FileNotFoundError(f"no image in {location}")
    if len(files) > 1:
        print(f"[input] {len(files)} files found; concatenating all")
    arrs = []
    for f in files:
        a = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(f))
        a = np.asarray(a)
        # 레이아웃 정규화 -> (N,H,W,3)
        if a.ndim == 3 and a.shape[-1] == 3:
            a = a[None]
        elif a.ndim == 3:                      # (N,H,W) 그레이
            a = np.repeat(a[..., None], 3, -1)
        elif a.ndim == 4 and a.shape[1] == 3:  # (N,3,H,W)
            a = np.moveaxis(a, 1, -1)
        arrs.append(a)
    stack = np.concatenate(arrs, axis=0)
    if stack.dtype != np.uint8:
        # [0,1] float 이면 255 스케일, 아니면 클립
        stack = np.clip(stack * 255 if stack.max() <= 1.5 else stack, 0, 255).astype(np.uint8)
    print(f"[input] stack shape={stack.shape} dtype={stack.dtype} "
          f"range=[{stack.min()},{stack.max()}]")
    return stack


# ---------------------------------------------------------------- 메인 핸들러
def interface_0_handler():
    import torch
    from rare_model import Ensemble, preprocess_frame
    _show_torch_cuda_info()

    cfg = load_config()
    ckpts = find_checkpoints()
    if not ckpts:
        # 안전장치: 가중치가 없으면 0.5 를 채워 컨테이너가 죽지 않게 (제출 전 반드시 교체)
        print("[FATAL] .pt 가중치를 찾지 못했습니다! MODEL_DIRS=", MODEL_DIRS)
        print("        model/ 또는 resources/ 에 best.pt 를 넣으세요.")

    stack = load_stack(INPUT_PATH / "images/stacked-barretts-esophagus-endoscopy")
    N = stack.shape[0]

    if not ckpts:
        out = [0.5] * N
        write_json_file(location=OUTPUT_PATH / "stacked-neoplastic-lesion-likelihoods.json",
                        content=out)
        return 0

    ens = Ensemble(ckpts, pool=cfg["pool"], topk_frac=cfg["topk_frac"],
                   mask_aware=cfg["mask_aware"], tta=cfg["tta"])

    # 이중백본 모드: config 에 dual 정보가 있으면 모델별 해상도로 전처리.
    # ckpts 순서(파일명 정렬)에 맞춰 각 모델의 size/weight 를 결정.
    dual = cfg.get("dual", None)
    bs = int(cfg["batch_size"])
    logits = np.empty(N, np.float32)
    import time
    t0 = time.time()

    if dual:
        # 파일명 접두사로 백본 판별 -> size/weight/tta 매핑
        def model_size_weight(path):
            fn = str(path).lower()
            # ── CAFormer (구체 prefix, wbclahe 체크보다 먼저) ──
            if "caformer" in fn or "caf" in fn:
                return (dual.get("caformer_size", 768),
                        dual.get("caformer_weight", 0.7),
                        dual.get("caformer_tta", 1))
            # ── wb+pauc 3백본 (구체적 prefix 우선 매칭) ──
            if "rn50wbclahe" in fn or "rn50wbp" in fn:
                return (dual.get("rn50wbp_size", dual["rn50_size"]),
                        dual.get("rn50wbp_weight", dual["rn50_weight"]),
                        dual.get("rn50wbp_tta", dual.get("rn50_tta", 1)))
            if "dinov3bwbclahe" in fn or "dinov3bwbp" in fn:
                return (dual.get("dv3wbp_size", dual["dinov3_size"]),
                        dual.get("dv3wbp_weight", dual["dinov3_weight"]),
                        dual.get("dv3wbp_tta", 1))
            if "dinov3wbclahe" in fn or "gwbp" in fn:
                # GastroDINO wb (+pauc). gwb_* 키 사용 (하위호환: 기존 하이브리드 A 도 이 분기).
                return (dual.get("gwb_size", dual["dinov3_size"]),
                        dual.get("gwb_weight", dual["dinov3_weight"]),
                        dual.get("gwb_tta", dual.get("dinov3_tta", 1)))
            if "wbclahe" in fn or "wb_clahe" in fn:
                # 일반 wb (구체 prefix 없는 경우) -> gwb 가중치 (하위호환)
                return (dual.get("gwb_size", dual["dinov3_size"]),
                        dual.get("gwb_weight", dual["dinov3_weight"]),
                        dual.get("gwb_tta", dual.get("dinov3_tta", 1)))
            if "vits" in fn:
                return (dual.get("vits_size", 768), dual.get("vits_weight", 0.4),
                        dual.get("vits_tta", 1))
            if "maxvit" in fn or "mvit" in fn:
                # MaxViT 는 512 전용 (768 window 비호환). 별도 크기/가중치.
                return (dual.get("maxvit_size", 512), dual.get("maxvit_weight", 0.5),
                        dual.get("maxvit_tta", 1))
            if "dinov3b" in fn:
                # 원본 DINOv3 (GastroDINO 와 별도 가중치). dinov3b_size 없으면 dinov3_size 상속.
                return (dual.get("dinov3b_size", dual["dinov3_size"]),
                        dual.get("dinov3b_weight", dual["dinov3_weight"]),
                        dual.get("dinov3b_tta", dual.get("dinov3_tta", 1)))
            if "dinov3" in fn or "dv3" in fn:
                return dual["dinov3_size"], dual["dinov3_weight"], dual.get("dinov3_tta", 1)
            return dual["rn50_size"], dual["rn50_weight"], dual.get("rn50_tta", 1)
        def model_cc(path):
            """파일명 기반 color_const. 하이브리드(원본+WB+CLAHE 혼합)용.
            파일명에 'wbclahe' 표식 있으면 wb_clahe, 없으면 전역 cfg color_const."""
            fn = str(path).lower()
            if "wbclahe" in fn or "wb_clahe" in fn:
                return "wb_clahe"
            return cfg["color_const"]
        sizes = [model_size_weight(p)[0] for p in ckpts]
        weights = [model_size_weight(p)[1] for p in ckpts]
        ttas = [model_size_weight(p)[2] for p in ckpts]
        ccs = [model_cc(p) for p in ckpts]
        print(f"[dual] fusion={dual.get('fusion','logit')}")
        print(f"[dual] 모델별 (size,weight,tta): "
              f"{list(zip([Path(p).name for p in ckpts], sizes, weights, ttas))}")
        for i in range(0, N, bs):
            chunk = list(stack[i:i + bs])            # 원본 RGB 프레임 그대로 전달
            logits[i:i + bs] = ens.logits_multiscale(
                chunk, sizes, weights, model_ttas=ttas,
                fov_crop=cfg["fov_crop"], color_const=cfg["color_const"],
                fusion=dual.get("fusion", "logit"), model_ccs=ccs)
            if i == 0 or (i // bs) % 20 == 0:
                done = min(i + bs, N)
                rate = done / max(time.time() - t0, 1e-6)
                print(f"[infer-dual] {done}/{N}  {rate:.1f} frame/s")
        # 배치적응 이상탐지 게이트 (config 에 gate 블록 있으면)
        gate = cfg.get("gate", None)
        if gate:
            try:
                feats = ens.dinov3_patch_features(
                    list(stack), gate.get("size", dual["dinov3_size"]),
                    fov_crop=cfg["fov_crop"], color_const=cfg["color_const"],
                    topk_frac=cfg["topk_frac"],
                    max_models=gate.get("max_models", 1))
                from rare_model import batch_adaptive_gate
                logits = batch_adaptive_gate(
                    logits, feats,
                    q=gate.get("q", 0.6), gate_lo=gate.get("gate_lo", 0.4),
                    penalty=gate.get("penalty", 2.0), shrink=gate.get("shrink", 0.1))
            except Exception as e:
                print(f"[gate] 실패 -> 게이트 없이 진행: {e}")
        _finalize(logits, N, t0)
        return 0

    for i in range(0, N, bs):
        chunk = stack[i:i + bs]
        batch = np.stack([preprocess_frame(f, cfg["size"], cfg["fov_crop"], cfg["color_const"])
                          for f in chunk])
        logits[i:i + bs] = ens.logits(batch)
        if i == 0 or (i // bs) % 20 == 0:
            done = min(i + bs, N)
            rate = done / max(time.time() - t0, 1e-6)
            print(f"[infer] {done}/{N}  {rate:.1f} frame/s  "
                  f"eta {(N-done)/max(rate,1e-6):.0f}s", flush=True)

    # sigmoid 한 번 (순위 보존). float 그대로 — round 금지.
    _finalize(logits, N, t0)
    return 0


def _finalize(logits, N, t0=None):
    import time
    probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    out = [float(p) for p in probs]
    el = f" in {time.time()-t0:.0f}s" if t0 else ""
    print(f"[infer] done {N} frames{el}  "
          f"prob range=[{min(out):.4f},{max(out):.4f}]  mean={np.mean(out):.4f}")
    write_json_file(location=OUTPUT_PATH / "stacked-neoplastic-lesion-likelihoods.json",
                    content=out)


# ---------------------------------------------------------------- GC 보일러플레이트
def run():
    interface_key = get_interface_key()
    handler = {("stacked-barretts-esophagus-endoscopy-images",): interface_0_handler}[interface_key]
    return handler()


def get_interface_key():
    inputs = load_json_file(location=INPUT_PATH / "inputs.json")
    return tuple(sorted(sv["interface"]["slug"] for sv in inputs))


def load_json_file(*, location):
    with open(location, "r") as f:
        return json.loads(f.read())


def write_json_file(*, location, content):
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))
    print(f"[output] wrote {location} ({len(content)} values)")


def _show_torch_cuda_info():
    import torch
    print("=+=" * 10)
    print(f"Torch CUDA available: {(av := torch.cuda.is_available())}")
    if av:
        print(f"\tdevices: {torch.cuda.device_count()}  "
              f"current: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
