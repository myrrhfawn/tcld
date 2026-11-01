import torch
import torch.nn as nn

from tcld.model import adabn


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4))
        self.encoder = nn.Sequential(nn.Conv2d(4, 4, 1), nn.BatchNorm2d(4))
        self.decoder = nn.Linear(4, 2)

    def forward(self, x):
        return self.decoder(self.encoder(self.backbone(x)).mean((2, 3)))


class _Adapter:
    def __init__(self):
        self.model = _Net()

    def set_norm_eval(self):
        for m in self.model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()


def test_recalibrate_updates_only_requested_parts_and_restores_mode():
    torch.manual_seed(0)
    a = _Adapter()
    a.model.train()
    a.set_norm_eval()
    bb, enc = a.model.backbone[1], a.model.encoder[1]
    enc_before = enc.running_mean.clone()
    x = [torch.randn(2, 3, 8, 8) * 3 + 5 for _ in range(4)]
    n = adabn.recalibrate(a, iter(x), parts=("backbone",))
    assert n == 4
    assert bb.running_mean.abs().sum() > 0 and int(bb.num_batches_tracked) == 4
    assert torch.equal(enc.running_mean, enc_before)
    assert a.model.training and not bb.training and bb.momentum == 0.1


def test_copy_bn_stats_matches_source():
    torch.manual_seed(1)
    src, dst = _Adapter(), _Adapter()
    adabn.recalibrate(src, iter([torch.randn(2, 3, 8, 8) + 2]))
    k = adabn.copy_bn_stats(src, dst)
    assert k == 2
    for ms, md in zip(adabn.bn_modules(src), adabn.bn_modules(dst)):
        assert torch.equal(ms.running_mean, md.running_mean) and torch.equal(ms.running_var, md.running_var)


def test_prior_blends_with_source_statistics():
    torch.manual_seed(2)
    a = _Adapter()
    bb = a.model.backbone[1]
    bb.running_mean.fill_(1.0)
    bb.running_var.fill_(4.0)
    x = [torch.zeros(2, 3, 8, 8)]          # target batch statistics: mean 0 after the conv bias only
    adabn.recalibrate(a, iter(x), parts=("backbone",), prior=1.0)
    assert torch.allclose(bb.running_mean, torch.ones(4)) and torch.allclose(bb.running_var, torch.full((4,), 4.0))
    adabn.recalibrate(a, iter(x), parts=("backbone",), prior=0.5)
    assert not torch.allclose(bb.running_mean, torch.ones(4))
