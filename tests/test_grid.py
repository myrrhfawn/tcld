import math

import pytest
import torch

from tcld.tcl import grid
from tcld.tcl.sharpness import cell_widths, differential_entropy, pixel_std, shannon_entropy

torch.manual_seed(0)


def _bin_values(reg_max=32, up=0.5, reg_scale=4.0):
    """Same construction as D-FINE weighting_function, in plain python."""
    ub1 = abs(up) * abs(reg_scale)
    ub2 = ub1 * 2
    step = (ub1 + 1) ** (2 / (reg_max - 2))
    left = [-(step**i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
    right = [step**i - 1 for i in range(1, reg_max // 2)]
    return torch.tensor([-ub2] + left + [0.0] + right + [ub2])


BV = _bin_values()
IMG = (640.0, 480.0)


def _soft_delta(R, idx, temp=0.3):
    x = torch.arange(R + 1, dtype=torch.float32)
    return torch.softmax(-((x - idx) ** 2) / temp, dim=-1)


def test_bin_values_match_upstream():
    from tcld.model.dfine_adapter import weighting_function

    up = torch.tensor([0.5])
    w = weighting_function(32, up, torch.tensor([4.0]))
    assert torch.allclose(w, BV, atol=1e-6)


def test_bin_positions_orientation_and_box_consistency():
    ref = torch.tensor([[0.5, 0.5, 0.2, 0.4]])  # cx, cy, w, h normalized
    pos = grid.bin_positions(ref, BV, 4.0, IMG)  # [1, 4, 33]
    assert pos.shape == (1, 4, 33)
    # left/top decrease with n, right/bottom increase
    assert (pos[0, 0, 1:] < pos[0, 0, :-1]).all() and (pos[0, 1, 1:] < pos[0, 1, :-1]).all()
    assert (pos[0, 2, 1:] > pos[0, 2, :-1]).all() and (pos[0, 3, 1:] > pos[0, 3, :-1]).all()
    # W(n)=0 bin (index 16) lies exactly on the anchor box edge
    x1 = (0.5 - 0.1) * 640
    x2 = (0.5 + 0.1) * 640
    y1 = (0.5 - 0.2) * 480
    assert math.isclose(pos[0, 0, 16].item(), x1, rel_tol=1e-5)
    assert math.isclose(pos[0, 2, 16].item(), x2, rel_tol=1e-5)
    assert math.isclose(pos[0, 1, 16].item(), y1, rel_tol=1e-5)


def test_resample_identity_same_grid():
    ref = torch.tensor([[0.4, 0.6, 0.3, 0.3]])
    pos = grid.bin_positions(ref, BV, 4.0, IMG)
    probs = torch.softmax(torch.randn(1, 4, 33), -1)
    out, kept = grid.resample(pos, probs, pos)
    assert torch.allclose(out, probs, atol=1e-5)
    assert torch.allclose(kept, torch.ones_like(kept), atol=1e-6)


def test_resample_shift_equivariance():
    """Shifting both the source grid and the target grid by the same amount yields
    identical masses (coordinates are absolute)."""
    ref_a = torch.tensor([[0.4, 0.5, 0.3, 0.3]])
    ref_b = torch.tensor([[0.45, 0.5, 0.25, 0.35]])
    pa = grid.bin_positions(ref_a, BV, 4.0, IMG)
    pb = grid.bin_positions(ref_b, BV, 4.0, IMG)
    probs = torch.softmax(torch.randn(1, 4, 33), -1)
    out1, _ = grid.resample(pa, probs, pb)
    out2, _ = grid.resample(pa + 17.0, probs, pb + 17.0)
    assert torch.allclose(out1, out2, atol=1e-6)


def test_resample_preserves_mass_and_mean_within_cell():
    ref_a = torch.tensor([[0.5, 0.5, 0.3, 0.3]])
    ref_b = torch.tensor([[0.5, 0.5, 0.36, 0.3]])  # wider anchor -> coarser grid, same centre
    pa = grid.bin_positions(ref_a, BV, 4.0, IMG)
    pb = grid.bin_positions(ref_b, BV, 4.0, IMG)
    probs = _soft_delta(32, 14.0, temp=4.0).expand(1, 4, 33).clone()
    out, kept = grid.resample(pa, probs, pb)
    assert torch.allclose(out.sum(-1), torch.ones(1, 4), atol=1e-5)
    assert (kept > 0.95).all()
    m_src, _ = grid.mean_var(pa, probs)
    m_dst, _ = grid.mean_var(pb, out)
    width = cell_widths(pb).max()
    assert (m_src - m_dst).abs().max() < width  # mean preserved up to one target cell


def test_resample_out_of_support_mass_reported():
    ref_a = torch.tensor([[0.5, 0.5, 0.3, 0.3]])
    ref_b = torch.tensor([[0.9, 0.5, 0.05, 0.3]])  # far away, tiny -> support does not overlap
    pa = grid.bin_positions(ref_a, BV, 4.0, IMG)
    pb = grid.bin_positions(ref_b, BV, 4.0, IMG)
    probs = _soft_delta(32, 16.0).expand(1, 4, 33).clone()
    _, kept = grid.resample(pa, probs, pb, renormalize=False)
    assert kept[0, 0] < 0.05  # left edge of a far-right box: no overlap


def test_resample_gradcheck():
    ref_a = torch.tensor([[0.5, 0.5, 0.3, 0.3]], dtype=torch.float64)
    ref_b = torch.tensor([[0.52, 0.49, 0.28, 0.33]], dtype=torch.float64)
    pa = grid.bin_positions(ref_a, BV.double(), 4.0, IMG)[:, :1]
    pb = grid.bin_positions(ref_b, BV.double(), 4.0, IMG)[:, :1]
    logits = torch.randn(1, 1, 33, dtype=torch.float64, requires_grad=True)

    def f(lg):
        return grid.resample(pa, torch.softmax(lg, -1), pb)[0]

    assert torch.autograd.gradcheck(f, (logits,), eps=1e-6, atol=1e-5)


def test_wasserstein_basic_properties():
    ref = torch.tensor([[0.5, 0.5, 0.3, 0.3]])
    pos = grid.bin_positions(ref, BV, 4.0, IMG)
    p = torch.softmax(torch.randn(1, 4, 33), -1)
    assert torch.allclose(grid.wasserstein1(pos, p, pos, p), torch.zeros(1, 4), atol=1e-6)
    # a delta shifted by 10 px on the same grid: W1 == 10
    d = torch.zeros(1, 1, 33)
    d[..., 16] = 1.0
    w = grid.wasserstein1(pos[:, :1], d, pos[:, :1] + 10.0, d)
    assert torch.allclose(w, torch.full((1, 1), 10.0), atol=1e-4)
    # symmetric
    q = torch.softmax(torch.randn(1, 4, 33), -1)
    assert torch.allclose(grid.wasserstein1(pos, p, pos, q), grid.wasserstein1(pos, q, pos, p), atol=1e-6)


def test_wasserstein_gradcheck():
    ref = torch.tensor([[0.5, 0.5, 0.3, 0.3]], dtype=torch.float64)
    pos = grid.bin_positions(ref, BV.double(), 4.0, IMG)[:, :1]
    la = torch.randn(1, 1, 33, dtype=torch.float64, requires_grad=True)
    lb = torch.randn(1, 1, 33, dtype=torch.float64)

    def f(x):
        return grid.wasserstein1(pos, torch.softmax(x, -1), pos + 3.0, torch.softmax(lb, -1))

    assert torch.autograd.gradcheck(f, (la,), eps=1e-6, atol=1e-5)


def test_shift_positions_with_scale():
    ref = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    pos = grid.bin_positions(ref, BV, 4.0, IMG)
    shifted = grid.shift_positions(pos, torch.tensor([[5.0, -3.0]]))
    assert torch.allclose(shifted[:, 0], pos[:, 0] + 5.0) and torch.allclose(shifted[:, 1], pos[:, 1] - 3.0)
    centre = torch.tensor([[320.0, 240.0]])
    scaled = grid.shift_positions(pos, torch.zeros(1, 2), torch.tensor([[2.0, 2.0]]), centre)
    assert torch.allclose(scaled[:, 2], 320.0 + (pos[:, 2] - 320.0) * 2.0)


def test_sharpness_differential_entropy_scales_with_box():
    """Same bin distribution on a 2x larger anchor is exactly log(2) nats less sharp."""
    probs = torch.softmax(torch.randn(1, 4, 33), -1)
    small = grid.bin_positions(torch.tensor([[0.5, 0.5, 0.1, 0.1]]), BV, 4.0, IMG)
    large = grid.bin_positions(torch.tensor([[0.5, 0.5, 0.2, 0.2]]), BV, 4.0, IMG)
    h_s = differential_entropy(small, probs)
    h_l = differential_entropy(large, probs)
    assert torch.allclose(h_l - h_s, torch.full_like(h_s, math.log(2.0)), atol=1e-4)
    # shannon entropy does not see the difference
    assert torch.allclose(shannon_entropy(probs), shannon_entropy(probs))
    assert torch.allclose(pixel_std(large, probs), 2 * pixel_std(small, probs), atol=1e-3)


@pytest.mark.parametrize("idx", [16.0, 3.0, 30.0])
def test_delta_is_sharper_than_spread(idx):
    pos = grid.bin_positions(torch.tensor([[0.5, 0.5, 0.2, 0.2]]), BV, 4.0, IMG)[:, :1]
    sharp = _soft_delta(32, idx, temp=0.2)[None, None]
    blurry = _soft_delta(32, idx, temp=20.0)[None, None]
    assert (differential_entropy(pos, sharp) < differential_entropy(pos, blurry)).all()
