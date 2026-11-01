"""TCLD criterion for one batch of unlabeled frame pairs (P4.6).

Pipeline per batch (B pairs):
  1. student forward on frames a and b (distribution mode) -> DistOutput each;
  2. per pair: confident queries, cross-frame matching (tcld.tcl.matching) with motion
     compensation from optical flow (default) / frozen model / predicted centres;
  3. bin positions in network-input pixels for matched queries, frame-a positions
     shifted into frame-b coordinates (+ optional scale);
  4. tcl_loss (asym | sym | ema) + anti-collapse terms (prediction anchor to the
     frozen model, L2-SP, entropy floor).
Everything in network-input pixel space (input_size), consistent with the flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torchvision

from tcld.data.common import COCO_TO_TARGET
from tcld.model.base import DistOutput
from tcld.model.dfine_adapter import DFINEAdapter
from tcld.tcl import collapse, grid
from tcld.tcl.loss import TCLConfig, tcl_loss
from tcld.tcl.matching import match_frames
from tcld.tcl.motion import motion_gate, scale_from_boxes, shift_from_boxes, shift_from_flow


@dataclass
class CriterionConfig:
    method: str = "tcld"             # tcld | coord | moments  (P5 baselines share matching / motion / gates)
    tcl: TCLConfig = field(default_factory=TCLConfig)
    tcl_weight: float = 1.0
    motion: str = "hybrid"           # hybrid (frozen→flow fallback) | flow | frozen | pred | none
    use_scale: bool = False          # also rescale frame-a positions by predicted size ratio
    score_thresh: float = 0.3
    iou_thresh: float = 0.5
    nms_iou: float = 0.7
    min_motion_px: float = 0.0       # drop pairs that barely move (static-background guard)
    max_motion_px: Optional[float] = None
    motion_weight: bool = False      # weight pairs by motion magnitude (sqrt)
    conf_weight: bool = True         # weight pairs by min(score_a, score_b)
    anchor_weight: float = 1.0       # prediction anchor to the frozen model
    anchor_iou: float = 0.7
    anchor_mode: str = "auto"        # auto (index when encoder frozen, else position) | index | position | iou
    ensemble_k: int = 1              # >1: temporal ensembling — target = mean of teacher distributions over k context frames
    motion_agree_px: Optional[float] = None   # keep pairs only if frozen-model shift and optical-flow shift agree within this (px)
    motion_agree_soft: bool = False  # instead of dropping: weight = exp(-disagreement / motion_agree_px)
    scale_source: str = "pred"       # pred | frozen — which boxes give the size ratio when use_scale
    all_layers: bool = False         # apply the TCL loss to every decoder layer (shared ref_points -> same bin positions)
    size_weight: float = 0.0         # translation-invariant term: W1 between width/height distributions (no motion needed)
    size_only: bool = False          # drop the positional (motion-compensated) TCL term, keep only the size term
    tcl_classes: Optional[List[int]] = None   # restrict TCL pairs to these target classes (0 person, 1 vehicle); None = all
    anchor_cls_weight: float = 1.0   # classification part of the anchor (BCE to frozen probs)
    anchor_box_weight: float = 1.0   # box part (GIoU + L1 to frozen boxes); 0 lets localisation move freely
    l2sp_weight: float = 0.0
    entropy_floor_weight: float = 0.0
    entropy_floor_h: float = 1.0


def target_scores(adapter: DFINEAdapter, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-query score / label restricted to the 2 target classes (person, vehicle)."""
    from src.data.dataset import mscoco_label2category

    idx = [i for i, c in mscoco_label2category.items() if c in COCO_TO_TARGET]
    cls = torch.tensor([COCO_TO_TARGET[mscoco_label2category[i]] for i in idx], device=logits.device)
    sub = logits.sigmoid()[..., idx]
    s, k = sub.max(-1)
    return s, cls[k]


class TCLDCriterion:
    def __init__(self, adapter: DFINEAdapter, frozen: Optional[DFINEAdapter], cfg: CriterionConfig,
                 input_size: Tuple[int, int], init_state: Optional[Dict[str, torch.Tensor]] = None):
        self.adapter = adapter
        self.frozen = frozen
        self.cfg = cfg
        self.input_size = input_size  # (h, w)
        self.init_state = init_state
        enc = adapter.model.encoder
        self.encoder_frozen = all(not p.requires_grad for p in enc.parameters()) and \
            all(not p.requires_grad for p in adapter.model.backbone.parameters())

    def _positions(self, d: DistOutput, b: int, q: torch.Tensor) -> torch.Tensor:
        W, H = self.input_size[1], self.input_size[0]
        return grid.bin_positions(d.ref_points[b][q], self.adapter.bin_values(), d.reg_scale, (W, H))

    def _boxes_px(self, d: DistOutput, b: int) -> torch.Tensor:
        W, H = self.input_size[1], self.input_size[0]
        return torchvision.ops.box_convert(d.boxes[b], "cxcywh", "xyxy") * torch.tensor([W, H, W, H], device=d.boxes.device)

    def __call__(self, da: DistOutput, db: DistOutput, flow: Optional[torch.Tensor],
                 fa: Optional[DistOutput] = None, fb: Optional[DistOutput] = None,
                 teacher_a: Optional[DistOutput] = None,
                 teacher_ctx: Optional[List[DistOutput]] = None,
                 frozen_ctx: Optional[List[DistOutput]] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.cfg
        B = da.logits.shape[0]
        if cfg.ensemble_k > 1 and teacher_ctx is not None and frozen_ctx is not None and fb is not None:
            return self._call_ensemble(da, db, fa, fb, teacher_ctx, frozen_ctx)
        sa, la = target_scores(self.adapter, da.logits)
        sb, lb = target_scores(self.adapter, db.logits)
        ca_list, pa_list, cb_list, pb_list, w_list, ta_list, tpa_list = [], [], [], [], [], [], []
        sa_list, spa_list, sb_list, spb_list, sw_list = [], [], [], [], []
        ext_list, sext_list = [], []
        n_matched, n_gated, n_agree_drop = 0, 0, 0
        cls_n = {0: [0, 0, 0.0], 1: [0, 0, 0.0]}   # per class: matched, passed motion gate, sum |shift| of passed
        for b in range(B):
            boxes_a, boxes_b = self._boxes_px(da, b).detach(), self._boxes_px(db, b).detach()
            # motion estimate for every frame-a query (used for matching and for compensation)
            if cfg.motion == "flow" and flow is not None:
                shift, valid = shift_from_flow(flow[b].float().cpu().numpy(), boxes_a.cpu())
                shift, valid = shift.to(boxes_a.device), valid.to(boxes_a.device)
            elif cfg.motion in ("frozen", "hybrid") and fa is not None and fb is not None:
                # frozen model boxes on both frames, matched by IoU: displacement of matched centres
                fba, fbb = self._boxes_px(fa, b), self._boxes_px(fb, b)
                fsa, fla = target_scores(self.adapter, fa.logits[b][None])
                fsb, flb = target_scores(self.adapter, fb.logits[b][None])
                mm = match_frames(fba, fsa[0], fla[0], fbb, fsb[0], flb[0], None, cfg.score_thresh, 0.3, cfg.nms_iou)
                shift = torch.zeros_like(boxes_a[:, :2])
                valid = torch.zeros(boxes_a.shape[0], dtype=torch.bool, device=boxes_a.device)
                if len(mm):
                    disp = shift_from_boxes(fba[mm.qa], fbb[mm.qb])
                    iou = torchvision.ops.box_iou(boxes_a, fba[mm.qa])
                    best, j = iou.max(-1)
                    ok = best >= 0.5
                    shift[ok] = disp[j[ok]]
                    valid = ok
                flow_shift = flow_valid = None
                if flow is not None and (cfg.motion == "hybrid" or cfg.motion_agree_px is not None):
                    fs, fv = shift_from_flow(flow[b].float().cpu().numpy(), boxes_a.cpu())
                    flow_shift, flow_valid = fs.to(boxes_a.device), fv.to(boxes_a.device)
                if cfg.motion == "hybrid" and flow_shift is not None and (~valid).any():
                    fill = (~valid) & flow_valid
                    shift[fill] = flow_shift[fill]
                    valid = valid | fill
            else:
                shift = torch.zeros_like(boxes_a[:, :2])
                valid = torch.ones(boxes_a.shape[0], dtype=torch.bool, device=boxes_a.device)
                flow_shift = flow_valid = None
            m = match_frames(boxes_a, sa[b], la[b], boxes_b, sb[b], lb[b],
                             shift if cfg.motion != "pred" else None, cfg.score_thresh, cfg.iou_thresh, cfg.nms_iou)
            if len(m) == 0:
                continue
            if cfg.motion == "pred":
                shift_m = shift_from_boxes(boxes_a[m.qa], boxes_b[m.qb])
                ok = torch.ones(len(m), dtype=torch.bool, device=boxes_a.device)
            else:
                shift_m = shift[m.qa]
                ok = valid[m.qa]
            gate = motion_gate(shift_m, cfg.min_motion_px, cfg.max_motion_px) & ok
            agree_w = None
            if cfg.motion_agree_px is not None and flow_shift is not None and cfg.motion != "pred":
                # reliability of the compensation: agreement of two independent motion estimates
                dis = (shift[m.qa] - flow_shift[m.qa]).norm(dim=-1)
                both = flow_valid[m.qa] & valid[m.qa]
                if cfg.motion_agree_soft:
                    agree_w = torch.where(both, torch.exp(-dis / cfg.motion_agree_px), torch.full_like(dis, 0.5))
                else:
                    gate = gate & both & (dis <= cfg.motion_agree_px)
                n_agree_drop += int((~(both & (dis <= cfg.motion_agree_px))).sum())
            lab_m = la[b][m.qa]
            mag = shift_m.norm(dim=-1)
            for c_ in (0, 1):
                sel = lab_m == c_
                cls_n[c_][0] += int(sel.sum())
                cls_n[c_][1] += int((sel & gate).sum())
                cls_n[c_][2] += float(mag[sel & gate].sum())
            if cfg.tcl_classes is not None:
                gate = gate & torch.isin(lab_m, torch.tensor(cfg.tcl_classes, device=lab_m.device))
            n_matched += len(m)
            n_gated += int((~gate).sum())
            if gate.sum() == 0:
                continue
            qa, qb, shift_m = m.qa[gate], m.qb[gate], shift_m[gate]
            if agree_w is not None:
                agree_w = agree_w[gate]
            pos_a = self._positions(da, b, qa)
            pos_b = self._positions(db, b, qb)
            # box extent per edge axis (w, h, w, h): mean of both frames, for the scale-invariant W1
            wh = 0.5 * ((boxes_a[qa][:, 2:] - boxes_a[qa][:, :2]) + (boxes_b[qb][:, 2:] - boxes_b[qb][:, :2]))
            extent = wh.repeat(1, 2)
            if cfg.size_weight > 0:
                sext_list.append(extent)
                sa_list.append(da.corners[b][qa])
                spa_list.append(pos_a)
                sb_list.append(db.corners[b][qb])
                spb_list.append(pos_b)
                sw_list.append(torch.minimum(sa[b][qa], sb[b][qb]).detach() if cfg.conf_weight else torch.ones(len(qa), device=pos_a.device))
            scale = centre = None
            if cfg.use_scale:
                if cfg.scale_source == "frozen" and fa is not None and fb is not None:
                    # size ratio of the frozen model's matched boxes (un-gameable); fallback to pred boxes
                    fba, fbb = self._boxes_px(fa, b), self._boxes_px(fb, b)
                    ia = torchvision.ops.box_iou(boxes_a[qa], fba).max(-1)
                    ib = torchvision.ops.box_iou(boxes_b[qb], fbb).max(-1)
                    scale = scale_from_boxes(boxes_a[qa], boxes_b[qb])
                    okf = (ia.values >= 0.5) & (ib.values >= 0.5)
                    scale[okf] = scale_from_boxes(fba[ia.indices[okf]], fbb[ib.indices[okf]])
                else:
                    scale = scale_from_boxes(boxes_a[qa], boxes_b[qb])
                scale = scale.clamp(0.5, 2.0)
                centre = (boxes_a[qa][:, :2] + boxes_a[qa][:, 2:]) / 2
            pos_a = grid.shift_positions(pos_a, shift_m, scale, centre)
            w = torch.ones(len(qa), device=boxes_a.device)
            if agree_w is not None:
                w = w * agree_w
            if cfg.conf_weight:
                w = w * torch.minimum(sa[b][qa], sb[b][qb]).detach()
            if cfg.motion_weight:
                w = w * shift_m.norm(dim=-1).clamp_min(1.0).sqrt()
            layers_a = [da.corners[b][qa]] + ([x.corners[b][qa] for x in da.aux] if cfg.all_layers else [])
            layers_b = [db.corners[b][qb]] + ([x.corners[b][qb] for x in db.aux] if cfg.all_layers else [])
            for la_, lb_ in zip(layers_a, layers_b):
                ca_list.append(la_)
                ext_list.append(extent)
                pa_list.append(pos_a)
                cb_list.append(lb_)
                pb_list.append(pos_b)
                w_list.append(w)
            if teacher_a is not None:
                ta_list.append(teacher_a.corners[b][qa].detach())
                tpa_list.append(grid.shift_positions(self._positions(teacher_a, b, qa), shift_m, scale, centre))
        stats: Dict[str, float] = {"n_matched": float(n_matched), "n_gated_out": float(n_gated), "n_agree_drop": float(n_agree_drop)}
        for c_, name in ((0, "person"), (1, "vehicle")):
            stats[f"n_matched_{name}"] = float(cls_n[c_][0])
            stats[f"n_pass_{name}"] = float(cls_n[c_][1])
            stats[f"shift_px_{name}"] = cls_n[c_][2] / max(cls_n[c_][1], 1)
        dev = da.logits.device
        if ca_list and cfg.size_only:
            ca_list = []   # positional term disabled
        if ca_list:
            if cfg.method == "tcld":
                out = tcl_loss(torch.cat(ca_list), torch.cat(pa_list), torch.cat(cb_list), torch.cat(pb_list), cfg.tcl,
                               weights=torch.cat(w_list),
                               target_corners=torch.cat(ta_list) if ta_list else None,
                               target_pos=torch.cat(tpa_list) if tpa_list else None,
                               extent=torch.cat(ext_list))
            else:
                out = baseline_consistency(cfg.method, torch.cat(ca_list), torch.cat(pa_list), torch.cat(cb_list),
                                           torch.cat(pb_list), torch.cat(w_list), cfg.tcl)
            loss = cfg.tcl_weight * out.loss
            stats.update({f"tcl_{k}": v for k, v in out.stats.items()})
            stats["loss_tcl"] = float(out.loss.detach())
        else:
            loss = torch.zeros((), device=dev, requires_grad=True)
            stats["loss_tcl"] = 0.0
        if cfg.size_weight > 0 and sa_list:
            from tcld.tcl.loss import size_consistency_loss

            so = size_consistency_loss(torch.cat(sa_list), torch.cat(spa_list), torch.cat(sb_list), torch.cat(spb_list),
                                       cfg.tcl, torch.cat(sw_list), mode="sym" if cfg.tcl.mode != "ema" else "ema",
                                       extent=torch.cat(sext_list))
            loss = loss + cfg.size_weight * so.loss
            stats["loss_size"] = float(so.loss.detach())
            stats["size_w1_mean"] = so.stats.get("size_w1_mean", 0.0)
        if cfg.anchor_weight > 0 and fa is not None:
            idx = cfg.anchor_mode == "index" or (cfg.anchor_mode == "auto" and self.encoder_frozen)
            pos = cfg.anchor_mode == "position" or (cfg.anchor_mode == "auto" and not self.encoder_frozen and da.topk_ind is not None)
            kw = dict(box_weight=cfg.anchor_box_weight, cls_weight=cfg.anchor_cls_weight, index_aligned=idx, position_aligned=pos)
            la_, st = collapse.prediction_anchor(da, fa, cfg.score_thresh, cfg.anchor_iou, **kw)
            lb_, _ = collapse.prediction_anchor(db, fb, cfg.score_thresh, cfg.anchor_iou, **kw) if fb is not None else (torch.zeros((), device=dev), {})
            stats["anchor_index_aligned"] = float(idx)
            stats["anchor_position_aligned"] = float(pos)
            stats["anchor_common_frac"] = st.get("anchor_common_frac", 1.0)
            anc = 0.5 * (la_ + lb_)
            loss = loss + cfg.anchor_weight * anc
            stats["loss_anchor"] = float(anc.detach())
        if cfg.l2sp_weight > 0 and self.init_state is not None:
            l2 = collapse.l2sp(self.adapter.model.named_parameters(), self.init_state)
            loss = loss + cfg.l2sp_weight * l2
            stats["loss_l2sp"] = float(l2.detach())
        if cfg.entropy_floor_weight > 0:
            W, H = self.input_size[1], self.input_size[0]
            pos = grid.bin_positions(db.ref_points, self.adapter.bin_values(), db.reg_scale, (W, H))
            ef = collapse.entropy_floor(pos, db.probs(), cfg.entropy_floor_h)
            loss = loss + cfg.entropy_floor_weight * ef
            stats["loss_entropy_floor"] = float(ef.detach())
        stats["loss_total"] = float(loss.detach())
        return loss, stats


def _call_ensemble(self, da, db, fa, fb, teacher_ctx, frozen_ctx):
    cfg = self.cfg
    dev = da.logits.device
    qb, tgt, stu_c, w, pos_b, stats = ensemble_target(self, db, teacher_ctx, frozen_ctx, fb, cfg)
    if qb.numel():
        loss = cfg.tcl_weight * ensemble_loss(tgt, stu_c, w, pos_b, cfg)
        stats["loss_tcl"] = float(loss.detach())
        stats["tcl_n_used"] = float(qb.numel())
    else:
        loss = torch.zeros((), device=dev, requires_grad=True)
        stats["loss_tcl"] = 0.0
        stats["tcl_n_used"] = 0.0
    if cfg.anchor_weight > 0 and fa is not None:
        idx = cfg.anchor_mode == "index" or (cfg.anchor_mode == "auto" and self.encoder_frozen)
        pos = cfg.anchor_mode == "position" or (cfg.anchor_mode == "auto" and not self.encoder_frozen and da.topk_ind is not None)
        kw = dict(box_weight=cfg.anchor_box_weight, cls_weight=cfg.anchor_cls_weight, index_aligned=idx, position_aligned=pos)
        la_, _ = collapse.prediction_anchor(da, fa, cfg.score_thresh, cfg.anchor_iou, **kw)
        lb_, _ = collapse.prediction_anchor(db, fb, cfg.score_thresh, cfg.anchor_iou, **kw)
        anc = 0.5 * (la_ + lb_)
        loss = loss + cfg.anchor_weight * anc
        stats["loss_anchor"] = float(anc.detach())
    stats["loss_total"] = float(loss.detach())
    return loss, stats


TCLDCriterion._call_ensemble = _call_ensemble


def baseline_consistency(method: str, corners_a, pos_a, corners_b, pos_b, weights, tcl_cfg: TCLConfig):
    """P5 baselines on the SAME matched pairs / compensated positions as TCLD:
    coord   — L1 between decoded edge positions (= expectation of the distribution)
              + GIoU between the decoded boxes, no distribution shape involved
              (jitter / coordinate-consistency loss).
    moments — L1 on mean AND on std of the edge distribution in pixels
              (uncertainty-aware point loss).
    Parity with tcl_loss (so that only the loss form differs): mode `sym` is the
    stop-gradient average 0.5 * (sg(a) -> b + a <- sg(b)) exactly as in tcl_loss, the
    per-edge term is clipped at `w1_cap` pixels before `w1_scale`, and the denominator is
    floored at 4 * `min_denom_pairs` edge weights. Mode `asym`: the sharper side teaches."""
    from tcld.tcl.loss import TCLOutput
    from tcld.tcl.sharpness import differential_entropy

    pa, pb = torch.softmax(corners_a, -1), torch.softmax(corners_b, -1)
    with torch.no_grad():
        h_a, h_b = differential_entropy(pos_a, pa), differential_entropy(pos_b, pb)
        a_teaches = h_a < h_b - tcl_cfg.sharp_margin
        b_teaches = h_b < h_a - tcl_cfg.sharp_margin
    m_a, v_a = grid.mean_var(pos_a, pa)
    m_b, v_b = grid.mean_var(pos_b, pb)
    if tcl_cfg.mode == "sym":
        # same value as |m_a - m_b| but each side receives half the gradient, as in tcl_loss
        l_mean = 0.5 * ((m_a.detach() - m_b).abs() + (m_a - m_b.detach()).abs())
        l_std = 0.5 * ((v_a.detach().sqrt() - v_b.sqrt()).abs() + (v_a.sqrt() - v_b.detach().sqrt()).abs())
    else:
        l_mean = torch.where(a_teaches, (m_a.detach() - m_b).abs(), torch.where(b_teaches, (m_a - m_b.detach()).abs(), torch.zeros_like(m_a)))
        l_std = torch.where(a_teaches, (v_a.detach().sqrt() - v_b.sqrt()).abs(), torch.where(b_teaches, (v_a.sqrt() - v_b.detach().sqrt()).abs(), torch.zeros_like(m_a)))
    use = (a_teaches | b_teaches) if tcl_cfg.mode != "sym" else torch.ones_like(a_teaches)
    loss_e = l_mean.clamp_max(tcl_cfg.w1_cap) / tcl_cfg.w1_scale
    if method == "moments":
        loss_e = loss_e + l_std.clamp_max(tcl_cfg.w1_cap) / tcl_cfg.w1_scale
    elif method == "coord":
        # GIoU on boxes rebuilt from the 4 edge means (xyxy = [left, top, right, bottom])
        box_a = torch.stack([m_a[:, 0], m_a[:, 1], m_a[:, 2], m_a[:, 3]], -1)
        box_b = torch.stack([m_b[:, 0], m_b[:, 1], m_b[:, 2], m_b[:, 3]], -1)
        if tcl_cfg.mode == "sym":
            g = 0.5 * ((1 - torchvision.ops.generalized_box_iou(box_a.detach(), box_b).diagonal())
                       + (1 - torchvision.ops.generalized_box_iou(box_a, box_b.detach()).diagonal()))
        else:
            g_ab = 1 - torchvision.ops.generalized_box_iou(box_a.detach(), box_b).diagonal()
            g_ba = 1 - torchvision.ops.generalized_box_iou(box_a, box_b.detach()).diagonal()
            g = torch.where(a_teaches.all(-1), g_ab, torch.where(b_teaches.all(-1), g_ba, torch.zeros_like(g_ab)))
        loss_e = loss_e + g[:, None] / 4
    else:
        raise ValueError(method)
    loss_e = torch.where(use, loss_e, torch.zeros_like(loss_e))
    denom = (use.float() * weights[:, None]).sum().clamp_min(max(1.0, 4.0 * tcl_cfg.min_denom_pairs))   # 4 edges per pair
    loss = (loss_e * weights[:, None]).sum() / denom
    return TCLOutput(loss, {"n_pairs": float(corners_a.shape[0]), "n_used": float(use.float().sum()),
                            "loss_mean_used": float(loss_e.detach().sum() / use.float().sum().clamp_min(1.0)),
                            "frac_a_teaches": float(a_teaches.float().mean())})


def ensemble_target(
    crit: "TCLDCriterion",
    db: DistOutput,
    teacher_ctx: List[DistOutput],
    frozen_ctx: List[DistOutput],
    fb: DistOutput,
    cfg: CriterionConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Temporal ensembling of the target (FINDINGS 2026-10-04: single-frame targets are as
    noisy as the student). For every context frame i (t-(k-1)s ... t):
      teacher detections on frame i  --(frozen-model displacement i->b)-->  frame-b coordinates,
      matched to the student's confident queries on frame b, resampled onto the student's bin
      grid; the per-query targets are averaged over the context frames that matched.
    Returns (qb indices [M], target probs on the student grid [M,4,R+1], student corners [M,4,R+1],
             weights [M], student bin positions [M,4,R+1]) concatenated over the batch, plus stats."""
    B = db.logits.shape[0]
    sb, lb = target_scores(crit.adapter, db.logits)
    out_q, out_t, out_c, out_w, out_p = [], [], [], [], []
    n_match_total, n_q_total = 0, 0
    for b in range(B):
        boxes_b = crit._boxes_px(db, b).detach()
        acc: Dict[int, List[Tuple[torch.Tensor, float]]] = {}
        for ti, fi in zip(teacher_ctx, frozen_ctx):
            st_i, lt_i = target_scores(crit.adapter, ti.logits[b][None])
            boxes_i = crit._boxes_px(ti, b).detach()
            # displacement i -> b from the frozen model (un-gameable), per teacher query
            fbi, fbb = crit._boxes_px(fi, b), crit._boxes_px(fb, b)
            fsi, fli = target_scores(crit.adapter, fi.logits[b][None])
            fsb, flb = target_scores(crit.adapter, fb.logits[b][None])
            mm = match_frames(fbi, fsi[0], fli[0], fbb, fsb[0], flb[0], None, cfg.score_thresh, 0.3, cfg.nms_iou)
            shift = torch.zeros_like(boxes_i[:, :2])
            valid = torch.zeros(boxes_i.shape[0], dtype=torch.bool, device=boxes_i.device)
            if len(mm):
                disp = shift_from_boxes(fbi[mm.qa], fbb[mm.qb])
                iou = torchvision.ops.box_iou(boxes_i, fbi[mm.qa])
                best, j = iou.max(-1)
                ok = best >= 0.5
                shift[ok] = disp[j[ok]]
                valid = ok
            m = match_frames(boxes_i, st_i[0], lt_i[0], boxes_b, sb[b], lb[b], shift, cfg.score_thresh, cfg.iou_thresh, cfg.nms_iou)
            if len(m) == 0:
                continue
            gate = motion_gate(m.shift, cfg.min_motion_px, cfg.max_motion_px) & valid[m.qa]
            if gate.sum() == 0:
                continue
            qa, qb, sh = m.qa[gate], m.qb[gate], m.shift[gate]
            pos_i = grid.shift_positions(crit._positions(ti, b, qa), sh)
            pos_b = crit._positions(db, b, qb)
            probs_i = torch.softmax(ti.corners[b][qa].detach() / cfg.tcl.temperature, -1)
            tgt, kept = grid.resample(pos_i, probs_i, pos_b, renormalize=cfg.tcl.renormalize)
            keep = (kept >= cfg.tcl.min_kept).all(-1)
            w_pair = torch.minimum(st_i[0][qa], sb[b][qb]).detach() if cfg.conf_weight else torch.ones(len(qa), device=qb.device)
            for k_ in torch.nonzero(keep).flatten().tolist():
                acc.setdefault(int(qb[k_]), []).append((tgt[k_], float(w_pair[k_])))
            n_match_total += int(keep.sum())
        if not acc:
            continue
        qs = sorted(acc)
        qs_t = torch.tensor(qs, device=db.logits.device)
        tg = torch.stack([torch.stack([x for x, _ in acc[q_]]).mean(0) for q_ in qs])
        ww = torch.tensor([sum(w for _, w in acc[q_]) / len(acc[q_]) for q_ in qs], device=tg.device)
        n_q_total += len(qs)
        out_q.append(qs_t)
        out_t.append(tg)
        out_c.append(db.corners[b][qs_t])
        out_w.append(ww)
        out_p.append(crit._positions(db, b, qs_t))
    stats = {"n_matched": float(n_match_total), "ens_queries": float(n_q_total),
             "ens_per_query": float(n_match_total / max(n_q_total, 1)), "n_gated_out": 0.0}
    if not out_q:
        z = torch.zeros(0, device=db.logits.device)
        return z.long(), z, z, z, z, stats
    return torch.cat(out_q), torch.cat(out_t), torch.cat(out_c), torch.cat(out_w), torch.cat(out_p), stats


def ensemble_loss(target_probs: torch.Tensor, student_corners: torch.Tensor, weights: torch.Tensor,
                  pos_b: torch.Tensor, cfg: CriterionConfig) -> torch.Tensor:
    """Target and student live on the SAME grid (pos_b): KL(target‖student) or W1."""
    from tcld.tcl.loss import _kl_to_student

    if cfg.tcl.divergence == "kl":
        loss_e = _kl_to_student(target_probs.detach(), student_corners, cfg.tcl.temperature)
    else:
        stu = torch.softmax(student_corners / cfg.tcl.temperature, -1)
        loss_e = grid.wasserstein1(pos_b, target_probs.detach(), pos_b, stu).clamp_max(cfg.tcl.w1_cap) / cfg.tcl.w1_scale
    denom = (weights[:, None].expand_as(loss_e)).sum().clamp_min(1.0)
    return (loss_e * weights[:, None]).sum() / denom
