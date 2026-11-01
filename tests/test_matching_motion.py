import numpy as np
import torch

from tcld.tcl import matching, motion


def _frame(boxes, scores, labels):
    return (torch.tensor(boxes, dtype=torch.float32), torch.tensor(scores, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long))


def test_match_frames_with_shift_and_class():
    # frame a: two persons + one vehicle + one low-score junk + one duplicate
    ba, sa, la = _frame([[10, 10, 50, 90], [200, 200, 240, 280], [300, 50, 400, 110], [0, 0, 5, 5], [11, 10, 51, 91]],
                        [0.9, 0.8, 0.95, 0.1, 0.85], [0, 0, 1, 0, 0])
    # frame b: everything moved by (+20, +5); vehicle class flips to person in b -> must not match
    bb, sb, lb = _frame([[30, 15, 70, 95], [220, 205, 260, 285], [320, 55, 420, 115]], [0.9, 0.8, 0.9], [0, 0, 0])
    shift = torch.tensor([[20.0, 5.0]] * 5)
    m = matching.match_frames(ba, sa, la, bb, sb, lb, shift_a=shift, score_thresh=0.3, iou_thresh=0.5)
    assert len(m) == 2
    assert sorted(m.qa.tolist()) == [0, 1] and sorted(m.qb.tolist()) == [0, 1]   # duplicate (4) suppressed by NMS
    assert (m.iou > 0.99).all() and torch.allclose(m.shift, torch.tensor([[20.0, 5.0]] * 2))
    # without compensation at delta this large, the IoU gate rejects the person pairs
    m0 = matching.match_frames(ba, sa, la, bb, sb, lb, shift_a=None, iou_thresh=0.5)
    assert len(m0) <= 1
    g = matching.match_frames(ba, sa, la, bb, sb, lb, shift_a=shift, method="greedy")
    assert sorted(g.qa.tolist()) == sorted(m.qa.tolist())


def test_match_frames_empty():
    ba, sa, la = _frame([[0, 0, 10, 10]], [0.1], [0])
    bb, sb, lb = _frame([[0, 0, 10, 10]], [0.9], [0])
    assert len(matching.match_frames(ba, sa, la, bb, sb, lb)) == 0


def test_shift_and_scale_from_boxes():
    a = torch.tensor([[0.0, 0.0, 10.0, 20.0]])
    b = torch.tensor([[5.0, 2.0, 25.0, 42.0]])
    assert torch.allclose(motion.shift_from_boxes(a, b), torch.tensor([[10.0, 12.0]]))
    assert torch.allclose(motion.scale_from_boxes(a, b), torch.tensor([[2.0, 2.0]]))


def test_flow_recovers_translation():
    rng = np.random.RandomState(0)
    H, W = 120, 160
    tex = rng.randint(0, 255, (H + 40, W + 40, 3), dtype=np.uint8)
    import cv2

    tex = cv2.GaussianBlur(tex, (5, 5), 1.5)
    a = tex[20:20 + H, 20:20 + W]
    dx, dy = 6, -3
    b = tex[20 - dy:20 - dy + H, 20 - dx:20 - dx + W]  # content moves by (+dx, +dy) in b
    flow = motion.dense_flow(a, b)
    boxes = torch.tensor([[30.0, 30.0, 100.0, 90.0], [0.0, 0.0, 2.0, 2.0]])
    shift, valid = motion.shift_from_flow(flow, boxes)
    assert valid.tolist() == [True, False]
    assert abs(shift[0, 0].item() - dx) < 1.0 and abs(shift[0, 1].item() - dy) < 1.0
    gate = motion.motion_gate(shift, min_px=2.0)
    assert gate[0] and not gate[1]
