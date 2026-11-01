import torch

from tcld.tcl import grid
from tcld.tcl.loss import TCLConfig, size_consistency_loss, size_distribution
from tests.test_grid import BV, IMG


def _dist(ref, temp_sharp=0.5):
    pos = grid.bin_positions(ref, BV, 4.0, IMG)
    x = torch.arange(33, dtype=torch.float32)
    logits = (-((x - 16.0) ** 2) / temp_sharp).expand(ref.shape[0], 4, 33).clone()
    return pos, logits


def test_size_distribution_is_translation_invariant():
    ref = torch.tensor([[0.5, 0.5, 0.2, 0.3]])
    pos, logits = _dist(ref)
    probs = torch.softmax(logits, -1)
    v1, m1 = size_distribution(pos, probs)
    v2, m2 = size_distribution(pos + 37.0, probs)      # whole object shifted
    assert torch.allclose(v1, v2, atol=1e-4) and torch.allclose(m1, m2)
    assert torch.allclose(m1.sum(-1), torch.ones(1, 2), atol=1e-5)
    mean_w = (v1[0, 0] * m1[0, 0]).sum()
    assert abs(mean_w.item() - 0.2 * 640) < 1.0      # expected width = anchor width when distribution is centred


def test_size_loss_zero_for_shifted_copy_and_positive_for_resized():
    ref_a = torch.tensor([[0.5, 0.5, 0.2, 0.3]])
    pos_a, la = _dist(ref_a)
    pos_b = pos_a + 25.0                                 # same object, moved
    lb = la.clone()
    la.requires_grad_(True)
    lb.requires_grad_(True)
    out = size_consistency_loss(la, pos_a, lb, pos_b, TCLConfig(divergence="w1", w1_scale=1.0))
    assert out.loss.item() < 1e-4
    ref_c = torch.tensor([[0.5, 0.5, 0.22, 0.3]])        # wider object (~13 px, below the W1 cap)
    pos_c, lc = _dist(ref_c)
    lc.requires_grad_(True)
    out2 = size_consistency_loss(la, pos_a, lc, pos_c, TCLConfig(divergence="w1", w1_scale=1.0))
    assert 3.0 < out2.loss.item() < 32.0                 # ~13 px width difference, below w1_cap
    out2.loss.backward()
    assert la.grad.abs().sum() > 0 and lc.grad.abs().sum() > 0   # sym: both sides learn
