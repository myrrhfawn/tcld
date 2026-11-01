import torch

from tcld.tcl import grid
from tcld.tcl.loss import TCLConfig, tcl_loss
from tests.test_grid import BV, IMG

torch.manual_seed(0)


def _pair(shift_px=0.0, sharp_a=True, M=3):
    ref_a = torch.tensor([[0.5, 0.5, 0.2, 0.3]]).repeat(M, 1)
    ref_b = torch.tensor([[0.52, 0.5, 0.21, 0.29]]).repeat(M, 1)
    pos_a = grid.bin_positions(ref_a, BV, 4.0, IMG) + shift_px
    pos_b = grid.bin_positions(ref_b, BV, 4.0, IMG)
    x = torch.arange(33, dtype=torch.float32)
    la = (-((x - 16.0) ** 2) / (0.5 if sharp_a else 30.0)).expand(M, 4, 33).clone()   # finite logits
    lb = (-((x - 16.0) ** 2) / (30.0 if sharp_a else 0.5)).expand(M, 4, 33).clone()
    return la.requires_grad_(True), pos_a, lb.requires_grad_(True), pos_b


def test_asym_direction_and_gradients():
    la, pa, lb, pb = _pair(sharp_a=True)
    out = tcl_loss(la, pa, lb, pb, TCLConfig(mode="asym", divergence="kl"))
    assert out.stats["frac_a_teaches"] == 1.0 and out.stats["n_used"] == 12
    out.loss.backward()
    assert la.grad is None or la.grad.abs().sum() == 0          # sharp side is the (detached) target
    assert lb.grad is not None and lb.grad.abs().sum() > 0      # blurry side learns
    la2, pa2, lb2, pb2 = _pair(sharp_a=False)
    out2 = tcl_loss(la2, pa2, lb2, pb2, TCLConfig(mode="asym"))
    assert out2.stats["frac_a_teaches"] == 0.0
    out2.loss.backward()
    assert la2.grad.abs().sum() > 0 and (lb2.grad is None or lb2.grad.abs().sum() == 0)


def test_sym_both_sides_get_gradient():
    la, pa, lb, pb = _pair()
    out = tcl_loss(la, pa, lb, pb, TCLConfig(mode="sym", divergence="kl"))
    out.loss.backward()
    assert la.grad.abs().sum() > 0 and lb.grad.abs().sum() > 0


def test_loss_decreases_as_student_approaches_target():
    la, pa, lb, pb = _pair(sharp_a=True)
    cfg = TCLConfig(mode="ema", divergence="kl")
    l0 = tcl_loss(la, pa, lb, pb, cfg, target_corners=la, target_pos=pa).loss.item()
    opt = torch.optim.Adam([lb], lr=0.2)
    for _ in range(50):
        opt.zero_grad()
        loss = tcl_loss(la, pa, lb, pb, cfg, target_corners=la, target_pos=pa).loss
        loss.backward()
        opt.step()
    l1 = tcl_loss(la, pa, lb, pb, cfg, target_corners=la, target_pos=pa).loss.item()
    assert l1 < 0.2 * l0
    # the student's decoded edge moved towards the teacher's absolute edge position
    mean_t, _ = grid.mean_var(pa, torch.softmax(la, -1))
    mean_s, _ = grid.mean_var(pb, torch.softmax(lb, -1))
    assert (mean_t - mean_s).abs().mean() < 2.0


def test_w1_mode_and_support_drop():
    la, pa, lb, pb = _pair()
    out = tcl_loss(la, pa, lb, pb, TCLConfig(mode="sym", divergence="w1", w1_scale=10.0))
    out.loss.backward()
    assert out.loss.item() > 0 and la.grad.abs().sum() > 0
    # shift frame-a positions far away: KL pairs get dropped for lack of support overlap
    la, pa, lb, pb = _pair(shift_px=400.0)
    out = tcl_loss(la, pa, lb, pb, TCLConfig(mode="sym", divergence="kl", min_kept=0.8))
    assert out.stats["frac_dropped_support"] > 0.9 and out.loss.item() == 0.0


def test_weights_and_empty():
    la, pa, lb, pb = _pair(M=4)
    w = torch.tensor([1.0, 0.0, 0.0, 0.0])
    out_w = tcl_loss(la, pa, lb, pb, TCLConfig(mode="sym"), weights=w)
    out_1 = tcl_loss(la[:1], pa[:1], lb[:1], pb[:1], TCLConfig(mode="sym"))
    assert torch.allclose(out_w.loss, out_1.loss, atol=1e-6)
    empty = tcl_loss(la[:0], pa[:0], lb[:0], pb[:0], TCLConfig())
    assert empty.loss.item() == 0.0 and empty.stats["n_pairs"] == 0


def test_scale_invariant_w1():
    # the same relative mismatch on a 4x larger box: pixel W1 is 4x larger, normalised W1 equal
    torch.manual_seed(0)
    la, lb = torch.randn(1, 4, 33), torch.randn(1, 4, 33)
    base = torch.linspace(-1, 1, 33).expand(1, 4, 33)
    cfg_px = TCLConfig(mode="sym", divergence="w1", w1_cap=1e9)
    cfg_n = TCLConfig(mode="sym", divergence="w1", w1_cap=1e9, size_ref=10.0, size_norm_down_only=False)
    small = tcl_loss(la, base * 10, lb, base * 10 + 1.0, cfg_n, extent=torch.full((1, 4), 20.0)).loss
    large = tcl_loss(la, base * 40, lb, base * 40 + 4.0, cfg_n, extent=torch.full((1, 4), 80.0)).loss
    assert torch.allclose(small, large, rtol=1e-4)
    px_s = tcl_loss(la, base * 10, lb, base * 10 + 1.0, cfg_px).loss
    px_l = tcl_loss(la, base * 40, lb, base * 40 + 4.0, cfg_px).loss
    assert torch.allclose(px_l, 4 * px_s, rtol=1e-4)
    # down-only: boxes smaller than size_ref keep the pixel loss; no extent -> unchanged
    cfg_d = TCLConfig(mode="sym", divergence="w1", w1_cap=1e9, size_ref=100.0)
    assert torch.allclose(tcl_loss(la, base * 10, lb, base * 10 + 1.0, cfg_d, extent=torch.full((1, 4), 20.0)).loss, px_s)
    assert torch.allclose(tcl_loss(la, base * 10, lb, base * 10 + 1.0, cfg_d).loss, px_s)


def test_min_denom_pairs():
    la, pa, lb, pb = _pair(M=4)
    base = tcl_loss(la, pa, lb, pb, TCLConfig(mode="sym", divergence="w1")).loss
    # 4 pairs, floor of 8 pairs -> half the loss; floor below the pair count -> unchanged
    assert torch.allclose(tcl_loss(la, pa, lb, pb, TCLConfig(mode="sym", divergence="w1", min_denom_pairs=8)).loss, base / 2)
    assert torch.allclose(tcl_loss(la, pa, lb, pb, TCLConfig(mode="sym", divergence="w1", min_denom_pairs=2)).loss, base)
