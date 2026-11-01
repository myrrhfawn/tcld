"""P5 baselines (coord / moments) must share the normalisation of the TCL loss so that the
comparison 'distribution vs coordinates' differs in the loss form only."""
import torch

from tcld.train.criterion import TCLConfig, baseline_consistency


def _pairs(M=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    ca = torch.randn(M, 4, 33, generator=g).requires_grad_(True)
    cb = torch.randn(M, 4, 33, generator=g).requires_grad_(True)
    base = torch.linspace(0, 64, 33)
    # left/top edges around 100 px, right/bottom edges around 250 px -> valid boxes
    offset = torch.tensor([100.0, 100.0, 250.0, 250.0])[None, :, None]
    pos_a = base[None, None, :].expand(M, 4, 33) + torch.rand(M, 4, 1, generator=g) * 20 + offset
    pos_b = pos_a + torch.rand(M, 4, 1, generator=g) * 6 - 3
    w = torch.ones(M)
    return ca, cb, pos_a, pos_b, w


def test_sym_gradient_to_both_sides_and_n0_floor():
    for method in ("coord", "moments"):
        ca, cb, pa, pb, w = _pairs()
        cfg0 = TCLConfig(mode="sym", w1_scale=10.0, min_denom_pairs=0.0)
        out0 = baseline_consistency(method, ca, pa, cb, pb, w, cfg0)
        assert torch.isfinite(out0.loss) and out0.loss > 0
        out0.loss.backward()
        assert ca.grad.abs().sum() > 0 and cb.grad.abs().sum() > 0
        ca2, cb2, pa2, pb2, w2 = _pairs()
        cfg32 = TCLConfig(mode="sym", w1_scale=10.0, min_denom_pairs=32.0)
        out32 = baseline_consistency(method, ca2, pa2, cb2, pb2, w2, cfg32)
        # 7 pairs x 4 edges = 28 edge weights < 4 * 32 -> loss scaled down by 28 / 128
        assert torch.isclose(out32.loss, out0.loss * 28.0 / 128.0, rtol=1e-4)


def test_sym_halves_gradient_relative_to_plain_l1():
    ca, cb, pa, pb, w = _pairs()
    cfg = TCLConfig(mode="sym", w1_scale=10.0)
    out = baseline_consistency("coord", ca, pa, cb, pb, w, cfg)
    out.loss.backward()
    g_sym = ca.grad.clone()
    # reference: the same loss without stop-gradient (pre-parity form)
    ca2, cb2, pa2, pb2, w2 = _pairs()
    from tcld.tcl import grid
    import torchvision
    m_a, _ = grid.mean_var(pa2, torch.softmax(ca2, -1))
    m_b, _ = grid.mean_var(pb2, torch.softmax(cb2, -1))
    giou = 1 - torchvision.ops.generalized_box_iou(m_a, m_b).diagonal()
    plain = ((m_a - m_b).abs().clamp_max(cfg.w1_cap) / cfg.w1_scale + giou[:, None] / 4).sum() / (7 * 4)
    assert torch.isclose(plain, out.loss, rtol=1e-5)   # same value
    plain.backward()
    assert torch.allclose(ca2.grad, 2 * g_sym, rtol=1e-4, atol=1e-7)   # half the gradient per side


def test_w1_cap_limits_one_bad_pair():
    ca, cb, pa, pb, w = _pairs()
    pb = pb.clone(); pb[0] += 500.0   # one wildly mismatched pair
    cfg = TCLConfig(mode="sym", w1_scale=10.0, w1_cap=32.0)
    out = baseline_consistency("coord", ca, pa, cb, pb, w, cfg)
    assert out.loss < (32.0 / 10.0 + 2.0 / 4) / 7 + 1.0
