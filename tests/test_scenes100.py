import numpy as np
from PIL import Image

from tcld.data import scenes100
from tcld.data.common import FrameReader


def test_list_cameras_from_frame_dirs(tmp_path, monkeypatch):
    monkeypatch.delenv("TCLD_CORRUPT", raising=False)
    root = tmp_path / "frames"
    for vid, n in (("003", 3), ("005", 1)):
        d = root / vid
        d.mkdir(parents=True)
        for i in range(n):
            Image.fromarray(np.full((8, 12, 3), i * 40, np.uint8)).save(d / f"{108000 + 6 * i:08d}.jpg")
    clips = scenes100.list_cameras(root=root)
    assert [c.camera for c in clips] == ["003"]          # a single frame is not a clip
    c = clips[0]
    assert c.num_frames == 3 and c.gt is None and abs(c.fps - 30.0 / 6) < 1e-6
    assert c.frame_path(2).name == "00108012.jpg"
    assert FrameReader(c).read(1).shape == (8, 12, 3)
    assert scenes100.list_cameras(ids=["005"], root=root) == []
