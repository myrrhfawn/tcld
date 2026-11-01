import numpy as np
import pytest

from tcld.data import corrupt


@pytest.mark.parametrize("kind", corrupt.KINDS)
def test_corrupt_deterministic_and_shape(kind):
    rgb = (np.random.default_rng(0).random((60, 90, 3)) * 255).astype(np.uint8)
    a = corrupt.apply(rgb, "MOT17-04", 7, kind)
    b = corrupt.apply(rgb, "MOT17-04", 7, kind)
    assert a.shape == rgb.shape and a.dtype == np.uint8
    assert np.array_equal(a, b)
    assert not np.array_equal(a, rgb)


def test_no_corruption_is_identity(monkeypatch):
    monkeypatch.delenv("TCLD_CORRUPT", raising=False)
    rgb = np.zeros((4, 4, 3), np.uint8)
    assert corrupt.active() == "" and corrupt.apply(rgb, "x", 0, corrupt.active()) is rgb
