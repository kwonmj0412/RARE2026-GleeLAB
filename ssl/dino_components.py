#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dino_components.py — DINO self-distillation 핵심 컴포넌트 (Step 32a)
================================================================
전체 학습 루프 전에 '가장 틀리기 쉬운 부분'을 격리 정의 + 검증.
  - DINOHead: projection head (MLP + weight-normed 마지막 층)
  - DINOLoss: teacher centering + temperature sharpening (collapse 방지의 핵심)
  - EMA teacher 업데이트

DINO collapse 방지 메커니즘 (정확히 구현해야 함):
  1. teacher 출력에 center 빼고 낮은 temp 로 sharpen -> 뾰족한 타겟
  2. student 는 높은 temp -> 부드러운 예측
  3. cross-entropy(teacher_sharp || student) 최소화
  4. center 는 teacher 출력의 EMA -> 특정 차원 지배 방지
  5. teacher = student 의 EMA (momentum 0.996+)

이 파일은 import 되어 dino_pretrain 에서 쓰임. 단독 실행시 자체 검증.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOHead(nn.Module):
    """DINO projection head: MLP(hidden) -> bottleneck -> weight-normed 마지막 층."""
    def __init__(self, in_dim, out_dim=8192, hidden_dim=2048, bottleneck_dim=256,
                 nlayers=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(nlayers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers += [nn.Linear(hidden_dim, bottleneck_dim)]
        self.mlp = nn.Sequential(*layers)
        # weight-normed 마지막 층 (norm 고정 = 1, DINO 표준)
        self.last = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last.weight_g.data.fill_(1)
        self.last.weight_g.requires_grad = False  # norm 고정

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)   # bottleneck L2 정규화
        return self.last(x)               # (B, out_dim) 로짓


class DINOLoss(nn.Module):
    """teacher centering + sharpening cross-entropy.
    collapse 방지의 핵심. center 는 EMA 로 갱신."""
    def __init__(self, out_dim=8192, teacher_temp=0.04, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.tt = teacher_temp
        self.st = student_temp
        self.cm = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_out, teacher_out):
        """student_out, teacher_out: list of (B, out_dim) — 각 crop 의 출력.
        student: 모든 crop, teacher: global crop 2개."""
        student = [s / self.st for s in student_out]
        # teacher: center 빼고 sharpen
        teacher = [F.softmax((t - self.center) / self.tt, dim=-1).detach()
                   for t in teacher_out]
        total, n_terms = 0.0, 0
        for ti, tq in enumerate(teacher):
            for si, sq in enumerate(student):
                if ti == si:
                    continue  # 같은 crop 끼리는 제외 (다른 뷰만 매칭)
                loss = torch.sum(-tq * F.log_softmax(sq, dim=-1), dim=-1)
                total = total + loss.mean()
                n_terms += 1
        self._update_center(torch.cat(teacher_out))
        return total / max(n_terms, 1)

    @torch.no_grad()
    def _update_center(self, teacher_out):
        batch_center = teacher_out.mean(0, keepdim=True)
        self.center = self.center * self.cm + batch_center * (1 - self.cm)


@torch.no_grad()
def ema_update(student, teacher, momentum):
    """teacher = momentum*teacher + (1-momentum)*student."""
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(momentum).add_(ps.data, alpha=1 - momentum)


# ================= 자체 검증 =================
if __name__ == "__main__":
    torch.manual_seed(0)
    B, in_dim, out_dim = 8, 1024, 8192
    head = DINOHead(in_dim, out_dim)
    loss_fn = DINOLoss(out_dim, teacher_temp=0.04, student_temp=0.1)

    # 2 global + 6 local crop 모사 (여기선 특징만)
    feats_student = [torch.randn(B, in_dim) for _ in range(8)]
    feats_teacher = [torch.randn(B, in_dim) for _ in range(2)]  # global 2개
    s_out = [head(f) for f in feats_student]
    t_out = [head(f) for f in feats_teacher]

    print("[검증 1] head 출력 형태")
    print(f"  student {len(s_out)}개 각 {s_out[0].shape}, teacher {len(t_out)}개")
    assert s_out[0].shape == (B, out_dim)

    print("\n[검증 2] loss 계산 (유한값, 양수)")
    l = loss_fn(s_out, t_out)
    print(f"  loss = {l.item():.4f}")
    assert torch.isfinite(l) and l > 0

    print("\n[검증 3] collapse 감지 — 모든 출력 동일하면?")
    same = head(torch.ones(B, in_dim))
    l_collapse = loss_fn([same]*8, [same]*2)
    print(f"  동일출력 loss = {l_collapse.item():.4f}")
    print("  (centering 이 작동하면 collapse 시에도 loss 가 0 으로 안 감)")

    print("\n[검증 4] center EMA 갱신")
    c0 = loss_fn.center.clone()
    for _ in range(5):
        loss_fn([head(torch.randn(B, in_dim)) for _ in range(8)],
                [head(torch.randn(B, in_dim)) for _ in range(2)])
    print(f"  center 변화량: {(loss_fn.center - c0).abs().mean().item():.6f} (>0 이어야)")
    assert (loss_fn.center - c0).abs().mean() > 0

    print("\n[검증 5] gradient 흐름")
    l = loss_fn([head(f) for f in feats_student], [head(f) for f in feats_teacher])
    l.backward()
    g = head.mlp[0].weight.grad
    print(f"  head 첫 층 grad norm: {g.norm().item():.4f} (>0 이어야)")
    assert g.norm() > 0

    print("\n[검증 6] weight_norm 마지막 층 norm 고정")
    print(f"  weight_g requires_grad: {head.last.weight_g.requires_grad} (False 여야)")
    assert not head.last.weight_g.requires_grad

    print("\n✅ DINO 핵심 컴포넌트 전부 검증 통과")
    
"""
python dino_components.py

[검증 1] head 출력 형태
  student 8개 각 torch.Size([8, 8192]), teacher 2개

[검증 2] loss 계산 (유한값, 양수)
  loss = 9.0354

[검증 3] collapse 감지 — 모든 출력 동일하면?
  동일출력 loss = 8.2724
  (centering 이 작동하면 collapse 시에도 loss 가 0 으로 안 감)

[검증 4] center EMA 갱신
  center 변화량: 0.007542 (>0 이어야)

[검증 5] gradient 흐름
  head 첫 층 grad norm: 0.3405 (>0 이어야)

[검증 6] weight_norm 마지막 층 norm 고정
  weight_g requires_grad: False (False 여야)

✅ DINO 핵심 컴포넌트 전부 검증 통과

"""
