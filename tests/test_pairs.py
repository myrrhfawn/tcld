import numpy as np
import torch
from PIL import Image

from tcld.data.common import VideoClip
from tcld.data.pairs import PairDataset, PairSampler, collate_pairs


def _jpg_clip(tmp_path, name, n=20, w=64, h=48):
    d = tmp_path / name
    d.mkdir()
    for f in range(n):
        arr = np.full((h, w, 3), f * 10 % 256, np.uint8)
        arr[:, : (f + 1), :] = 255  # white stripe whose width encodes the frame index
        Image.fromarray(arr).save(d / f"{f + 1:06d}.jpg", quality=100)
    return VideoClip(name, "cam", None, d, n, 30.0, w, h)


def test_sampler_deterministic_and_bounded(tmp_path):
    clips = [_jpg_clip(tmp_path, "a", n=100), _jpg_clip(tmp_path, "b", n=7)]
    s1 = PairSampler(clips, deltas=(1, 5), base_stride=10, seed=3)
    s2 = PairSampler(clips, deltas=(1, 5), base_stride=10, seed=3)
    assert [p for p in s1.pairs] == [p for p in s2.pairs]
    for p in s1.pairs:
        clip = clips[p.clip_idx]
        assert 0 <= p.t and p.t + p.delta < clip.num_frames
        assert p.delta in (1, 5)
    assert len(s1) == 10 + 1  # a: t in 0..90 step 10 (max_d=5 -> stop at 95) ; b: only t=0
    s3 = PairSampler(clips, deltas=(1,), base_stride=1, max_pairs_per_clip=5, seed=0)
    assert sum(1 for p in s3.pairs if p.clip_idx == 0) == 5


def test_dataset_reads_correct_frames(tmp_path):
    clip = _jpg_clip(tmp_path, "a", n=20)
    s = PairSampler([clip], deltas=(3,), base_stride=4, seed=0)
    ds = PairDataset(s, input_size=(48, 64))
    item = ds[1]
    assert item["meta"]["t"] == 4 and item["meta"]["delta"] == 3
    a, b = ds.load_pair(s[1])
    assert (a[:, :5, :] > 200).all() and (a[:, 6:, :] < 120).all()      # frame 4 -> 5 white columns
    assert (b[:, :8, :] > 200).all() and (b[:, 9:, :] < 120).all()      # frame 7 -> 8 white columns
    assert item["img_a"].shape == (3, 48, 64) and item["img_a"].dtype == torch.float32
    assert item["meta"]["orig_size"] == (64, 48)


def test_photometric_is_shared_and_deterministic(tmp_path):
    clip = _jpg_clip(tmp_path, "a", n=10)
    s = PairSampler([clip], deltas=(1,), base_stride=5, seed=0)
    ds1 = PairDataset(s, input_size=(48, 64), photometric=0.4, seed=1)
    ds2 = PairDataset(s, input_size=(48, 64), photometric=0.4, seed=1)
    x1, x2 = ds1[0], ds2[0]
    assert torch.allclose(x1["img_a"], x2["img_a"]) and torch.allclose(x1["img_b"], x2["img_b"])
    # same photometric factors for both frames: identical constant-background pixels get identical values
    bg1 = x1["img_a"][:, -1, -1]
    bg2 = x1["img_b"][:, -1, -1]
    # frames 0 and 1 have backgrounds 0 and 10 -> not identical, but transform is monotone: b brighter than a
    assert (bg2 >= bg1).all()
    batch = collate_pairs([ds1[0], ds1[1]])
    assert batch["img_a"].shape == (2, 3, 48, 64) and len(batch["meta"]) == 2
