import torch


def test_eval_mode_outputs_boxes_only(dfine_s_random):
    """Eval mode caches anchors / positional embeddings for `eval_spatial_size`
    (640x640 upstream), so eval inputs must have exactly that size. Train /
    distribution mode recomputes them per input and accepts any size."""
    m = dfine_s_random
    x = torch.rand(1, 3, 640, 640)
    m.model.eval()
    out = m.model(x)
    assert set(out.keys()) == {"pred_logits", "pred_boxes"}
    assert out["pred_boxes"].shape == (1, 300, 4)


def test_distribution_mode_shapes_and_restoration(dfine_s_random):
    m = dfine_s_random
    m.model.eval()
    dn_before = m.decoder.num_denoising
    x = torch.rand(2, 3, 320, 320)
    with torch.no_grad():
        d = m.forward_distributions(x, all_layers=True)
    R = m.reg_max
    assert d.corners.shape == (2, 300, 4, R + 1)
    assert d.ref_points.shape == (2, 300, 4)
    assert d.boxes.shape == (2, 300, 4)
    assert d.logits.shape[:2] == (2, 300)
    assert len(d.aux) == m.decoder.num_layers - 1
    # mode restored: eval again, denoising setting back
    assert not m.model.training
    assert m.decoder.num_denoising == dn_before


def test_corners_to_boxes_identity(dfine_s_random):
    """Boxes emitted by the decoder must equal Integral(softmax(corners)) decoded
    against ref_points — this is the contract tcld.tcl.grid relies on."""
    m = dfine_s_random
    x = torch.rand(1, 3, 320, 320)
    with torch.no_grad():
        d = m.forward_distributions(x, all_layers=True)
        rec = m.boxes_from_corners(d.corners, d.ref_points)
        assert torch.allclose(rec, d.boxes, atol=1e-5), (rec - d.boxes).abs().max()
        for aux in d.aux:
            rec = m.boxes_from_corners(aux.corners, aux.ref_points)
            assert torch.allclose(rec, aux.boxes, atol=1e-5)


def test_batchnorm_stays_eval_in_distribution_mode(dfine_s_random):
    m = dfine_s_random
    m.model.eval()
    bns = [mod for mod in m.model.modules() if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm)]
    assert bns, "expected BatchNorm layers in D-FINE-S"
    with m.distribution_mode():
        assert m.model.training
        assert all(not bn.training for bn in bns)


def test_freeze(dfine_s_random):
    m = dfine_s_random
    m.freeze(["backbone", "encoder"])
    assert all(not p.requires_grad for p in m.model.backbone.parameters())
    assert all(not p.requires_grad for p in m.model.encoder.parameters())
    assert any(p.requires_grad for p in m.model.decoder.parameters())
    n_trainable = sum(p.numel() for p in m.trainable_parameters())
    assert 0 < n_trainable < sum(p.numel() for p in m.model.parameters())


def test_gradients_reach_decoder(dfine_s_random):
    m = dfine_s_random
    m.freeze(["backbone", "encoder"])
    x = torch.rand(1, 3, 320, 320)
    d = m.forward_distributions(x)
    loss = torch.softmax(d.corners, -1).var()
    loss.backward()
    grads = [p.grad for p in m.model.decoder.parameters() if p.requires_grad and p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
