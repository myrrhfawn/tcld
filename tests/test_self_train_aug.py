"""Mean-Teacher student view: flip maps cx -> 1 - cx, labels untouched, shapes preserved."""
import random

import torch

from tcld.train.self_train import augment_student_views


def _batch():
    imgs = torch.rand(2, 3, 32, 48)
    tgts = [{"labels": torch.tensor([1, 3]), "boxes": torch.tensor([[0.2, 0.5, 0.1, 0.2], [0.7, 0.3, 0.2, 0.1]])},
            {"labels": torch.zeros(0, dtype=torch.long), "boxes": torch.zeros(0, 4)}]
    return imgs, tgts


def test_flip_maps_cx():
    imgs, tgts = _batch()
    flipped = False
    for seed in range(20):
        rng = random.Random(seed)
        out, otg = augment_student_views(imgs, tgts, rng, photometric=0.0, hflip=True)
        assert out.shape == imgs.shape
        assert torch.equal(otg[0]["labels"], tgts[0]["labels"])
        if torch.equal(out[0], imgs[0].flip(-1)):
            flipped = True
            assert torch.allclose(otg[0]["boxes"][:, 0], 1.0 - tgts[0]["boxes"][:, 0])
            assert torch.allclose(otg[0]["boxes"][:, 1:], tgts[0]["boxes"][:, 1:])
        else:
            assert torch.equal(out[0], imgs[0])
            assert torch.equal(otg[0]["boxes"], tgts[0]["boxes"])
        assert otg[1]["boxes"].shape == (0, 4)
    assert flipped
    assert torch.equal(tgts[0]["boxes"], torch.tensor([[0.2, 0.5, 0.1, 0.2], [0.7, 0.3, 0.2, 0.1]]))  # input untouched


def test_photometric_keeps_range_and_boxes():
    imgs, tgts = _batch()
    out, otg = augment_student_views(imgs, tgts, random.Random(0), photometric=0.5, hflip=False)
    assert out.shape == imgs.shape and out.min() >= 0 and out.max() <= 1
    assert not torch.equal(out, imgs)
    assert torch.equal(otg[0]["boxes"], tgts[0]["boxes"])
