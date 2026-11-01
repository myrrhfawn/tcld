import torch

from tcld.model.base import DistOutput
from tcld.tcl import collapse


def _out(logits, boxes, topk_ind=None, enc_logits=None):
    B, Q = logits.shape[:2]
    return DistOutput(logits=logits, boxes=boxes, corners=torch.zeros(B, Q, 4, 33), ref_points=boxes.clone(),
                      up=torch.tensor(0.5), reg_scale=torch.tensor(4.0), topk_ind=topk_ind, enc_logits=enc_logits)


def test_position_anchor_matches_index_anchor_on_permuted_queries():
    torch.manual_seed(0)
    B, Q, C, A = 2, 6, 5, 40
    logits = torch.randn(B, Q, C) * 3
    boxes = torch.rand(B, Q, 4) * 0.5 + 0.25
    ind = torch.stack([torch.randperm(A)[:Q] for _ in range(B)])
    perm = torch.stack([torch.randperm(Q) for _ in range(B)])
    student = _out(logits.clone().requires_grad_(True), boxes.clone(), topk_ind=ind)
    # the frozen model selected the same positions in another order -> identical content, permuted
    f_logits = torch.stack([logits[b][perm[b]] for b in range(B)])
    f_boxes = torch.stack([boxes[b][perm[b]] for b in range(B)])
    f_ind = torch.stack([ind[b][perm[b]] for b in range(B)])
    frozen = _out(f_logits, f_boxes, topk_ind=f_ind, enc_logits=torch.zeros(B, A, C))
    l_pos, st = collapse.prediction_anchor(student, frozen, position_aligned=True)
    l_idx, _ = collapse.prediction_anchor(_out(logits.clone(), boxes.clone()), _out(logits.clone(), boxes.clone()), index_aligned=True)
    assert st["anchor_common_frac"] == 1.0
    assert torch.allclose(l_pos, l_idx, atol=1e-5)
    l_pos.backward()
    assert student.logits.grad is not None


def test_position_anchor_falls_back_to_encoder_logits_for_new_queries():
    torch.manual_seed(1)
    B, Q, C, A = 1, 4, 3, 10
    student = _out(torch.randn(B, Q, C).requires_grad_(True), torch.rand(B, Q, 4) * 0.5 + 0.25,
                   topk_ind=torch.tensor([[1, 2, 3, 4]]))
    enc = torch.randn(B, A, C)
    frozen = _out(torch.randn(B, Q, C) * 5, torch.rand(B, Q, 4) * 0.5 + 0.25,
                  topk_ind=torch.tensor([[3, 1, 8, 9]]), enc_logits=enc)
    loss, st = collapse.prediction_anchor(student, frozen, position_aligned=True)
    assert st["anchor_common_frac"] == 0.5          # positions 1 and 3 are shared
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert student.logits.grad.abs().sum(-1).gt(0).all()   # every query, shared or not, receives an anchor gradient
