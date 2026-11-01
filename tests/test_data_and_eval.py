import numpy as np
import pytest

from tcld.data import mot, virat
from tcld.data.common import FrameGT, VideoClip, split_clips_by_camera
from tcld.eval.two_class import Detections, apg, coco_dets_to_target, evaluate_camera, filter_ignore


def test_virat_clip_id_parsing():
    m = virat.parse_clip_id("VIRAT_S_010203_09_001010_001036")
    assert m == {"scene": "0102", "group": "03", "segment": "09", "start": "001010", "end": "001036"}
    assert virat.parse_clip_id("VIRAT_S_000001")["scene"] == "0000"
    with pytest.raises(ValueError):
        virat.parse_clip_id("MOT17-02")


def test_virat_objects_to_gt_mapping(tmp_path):
    p = tmp_path / "x.viratdata.objects.txt"
    p.write_text(
        "0 10 5 100 100 20 40 1\n"   # person
        "1 10 5 200 200 60 30 2\n"   # car -> vehicle
        "2 10 5 300 300 60 30 3\n"   # other vehicle -> vehicle
        "3 10 5 400 400 10 10 4\n"   # object -> ignore
        "4 10 6 50 50 10 10 5\n"     # bike -> ignore (frame 6)
    )
    gt = virat.objects_to_gt(virat.load_objects(p))
    assert set(gt) == {5, 6}
    f5 = gt[5]
    assert f5.boxes.shape == (3, 4) and f5.labels.tolist() == [0, 1, 1]
    assert f5.boxes[0].tolist() == [100, 100, 120, 140]
    assert f5.ignore_boxes.shape == (1, 4) and f5.track_ids.tolist() == [0, 1, 2]
    assert gt[6].boxes.shape == (0, 4) and gt[6].ignore_boxes.shape == (1, 4)


def test_mot_gt_parsing(tmp_path):
    p = tmp_path / "gt.txt"
    p.write_text(
        "1,1,10,10,20,50,1,1,1.0\n"    # pedestrian, evaluated
        "1,2,30,30,20,50,0,1,0.5\n"    # pedestrian, conf 0 -> ignore
        "1,3,60,60,20,50,1,7,1.0\n"    # static person -> person
        "1,4,90,90,20,50,1,3,1.0\n"    # car -> dropped
        "1,5,120,120,20,50,1,8,1.0\n"  # distractor -> ignore
        "2,1,11,10,20,50,1,1,1.0\n"
    )
    gt = mot.load_gt(p)
    assert set(gt) == {0, 1}
    assert gt[0].boxes.shape == (2, 4) and gt[0].track_ids.tolist() == [1, 3]
    assert gt[0].ignore_boxes.shape == (2, 4)
    assert gt[1].boxes[0].tolist() == [11, 10, 31, 60]


def _clip(cid, cam):
    return VideoClip(cid, cam, None, None, 10, 30.0, 100, 100)


def test_split_is_deterministic_and_per_camera():
    clips = [_clip(f"A{i}", "A") for i in range(10)] + [_clip(f"B{i}", "B") for i in range(3)] + [_clip("C0", "C")]
    s1 = split_clips_by_camera(clips, eval_frac=0.3, seed=0)
    s2 = split_clips_by_camera(clips, eval_frac=0.3, seed=0)
    assert [c.clip_id for c in s1["A"][1]] == [c.clip_id for c in s2["A"][1]]
    assert len(s1["A"][1]) == 3 and len(s1["A"][0]) == 7
    assert len(s1["B"][1]) == 1 and len(s1["B"][0]) == 2
    assert len(s1["C"][1]) == 0 and len(s1["C"][0]) == 1  # single clip: nothing to hold out
    assert not set(c.clip_id for c in s1["A"][0]) & set(c.clip_id for c in s1["A"][1])
    # a camera's split must not depend on which other cameras are present
    s_alone = split_clips_by_camera([c for c in clips if c.camera == "A"], eval_frac=0.3, seed=0)
    assert [c.clip_id for c in s_alone["A"][1]] == [c.clip_id for c in s1["A"][1]]


def test_coco_dets_mapping_and_ignore_filter():
    boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]], np.float32)
    d = coco_dets_to_target(boxes, np.array([0.9, 0.8, 0.7]), np.array([1, 6, 17]))  # person, bus, cat
    assert d.labels.tolist() == [0, 1] and d.boxes.shape == (2, 4)
    gt = FrameGT(boxes=np.zeros((0, 4), np.float32), labels=np.zeros(0, np.int64),
                 track_ids=np.zeros(0, np.int64), ignore_boxes=np.array([[18, 18, 32, 32]], np.float32))
    keep = filter_ignore(d.boxes, gt)
    assert keep.tolist() == [True, False]
    mask = np.zeros((60, 60), bool)
    mask[0:12, 0:12] = True
    assert filter_ignore(d.boxes, gt, mask).tolist() == [False, False]


def test_evaluate_camera_perfect_and_weighted():
    rng = np.random.RandomState(0)
    frames = []
    for _ in range(4):
        b = np.array([[10, 10, 50, 90], [100, 100, 180, 140], [200, 50, 260, 90]], np.float32)
        labels = np.array([0, 1, 1])
        gt = FrameGT(b, labels, np.arange(3))
        dets = Detections(b + rng.uniform(-0.05, 0.05, b.shape).astype(np.float32), np.array([0.9, 0.8, 0.7], np.float32), labels)
        frames.append((gt, dets))
    r = evaluate_camera(frames)
    assert r["APm_co"] > 99 and r["AP50_co"] > 99 and r["n_gt"] == 12
    assert abs(r["APm_w"] - r["APm_co"]) < 1e-6  # both classes perfect -> weighting irrelevant
    # drop all vehicle detections: class-mean halves, weighted drops by 2/3
    frames2 = [(g, Detections(d.boxes[:1], d.scores[:1], d.labels[:1])) for g, d in frames]
    r2 = evaluate_camera(frames2)
    assert abs(r2["APm_co"] - r["APm_co"] / 2) < 1.0
    assert abs(r2["APm_w"] - r["APm_w"] / 3) < 1.0


def test_apg():
    base = {"c1": {"APm_co": 40.0, "AP50_co": 60.0}, "c2": {"APm_co": 50.0, "AP50_co": 70.0}, "c3": {"APm_co": 1.0}}
    adapted = {"c1": {"APm_co": 43.0, "AP50_co": 64.0}, "c2": {"APm_co": 51.0, "AP50_co": 70.0}}
    g = apg(base, adapted, keys=("APm_co", "AP50_co"))
    assert g["n_cameras"] == 2 and abs(g["APG_APm_co"] - 2.0) < 1e-9 and abs(g["APG_AP50_co"] - 2.0) < 1e-9


def test_mot_list_sequences_split_auto_and_test(tmp_path):
    """Cross-recording protocol: a test sequence (no gt) loads via split='auto' and carries gt=None."""
    import configparser
    from tcld.data import mot
    for split, name, with_gt in (("train", "MOT17-04", True), ("test", "MOT17-03", False)):
        d = tmp_path / "MOT17Det" / split / name
        (d / "img1").mkdir(parents=True)
        cp = configparser.ConfigParser(); cp["Sequence"] = {"name": name, "imDir": "img1", "frameRate": "30", "seqLength": "10", "imWidth": "64", "imHeight": "48", "imExt": ".jpg"}
        with open(d / "seqinfo.ini", "w") as f: cp.write(f)
        if with_gt:
            (d / "gt").mkdir(); (d / "gt" / "gt.txt").write_text("1,1,10,10,5,5,1,1,1.0\n")
    tr = mot.list_sequences(root=tmp_path, names=["MOT17-04"])
    te = mot.list_sequences(root=tmp_path, names=["MOT17-03"], with_gt=False)
    assert tr[0].gt is not None and len(tr[0].gt) == 1 and tr[0].num_frames == 10
    assert te[0].gt is None and te[0].num_frames == 10
    assert mot.list_sequences(root=tmp_path, names=["MOT17-03"], split="train") == []
    assert mot.SAME_CAMERA_TEST["MOT17-04"] == "MOT17-03"
