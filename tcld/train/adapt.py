"""Per-camera self-supervised adaptation loop (P4.6).

  adapt_camera(exp_cfg, profile, camera, train_clips, eval_clips, seed) ->
      {final metrics, diagnostics history, checkpoint path}

Steps per iteration: one batch of frame pairs (unlabeled) -> student forward on a
and b in distribution mode -> frozen model forward (no grad) for motion/anchor ->
TCLDCriterion -> backward -> optimizer (AdamW on trainable params) -> EMA update.
Diagnostics (tcld.tcl.collapse.diagnose) on a fixed probe batch every `diag_every`
iterations; evaluation on the camera's eval clips every `eval_every` iterations and
at the end (both student and EMA; the better of the two at the end is reported as
`final`, the choice is recorded).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from tcld.data.common import VideoClip
from tcld.data.pairs import PairDataset, PairSampler, collate_pairs, worker_init
from tcld.eval.run import evaluate_cameras
from tcld.model.dfine_adapter import DFINEAdapter, load_dfine
from tcld.model.ema import ModelEMA
from tcld.tcl import collapse
from tcld.tcl.loss import TCLConfig
from tcld.train.criterion import CriterionConfig, TCLDCriterion


def build_criterion_cfg(d: dict) -> CriterionConfig:
    d = dict(d)
    tcl = TCLConfig(**d.pop("tcl", {}))
    return CriterionConfig(tcl=tcl, **d)


def adapt_camera(
    exp: dict,
    profile: dict,
    camera: str,
    train_clips: List[VideoClip],
    eval_clips: List[VideoClip],
    seed: int,
    out_dir: Path,
    eval_start_frac: float = 0.0,
    log=print,
) -> Dict:
    torch.manual_seed(seed)
    device = profile.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    input_size = tuple(profile.get("input_size", [640, 640]))
    amp = bool(profile.get("amp", True)) and device == "cuda"
    out_dir.mkdir(parents=True, exist_ok=True)

    # models
    model_name = exp.get("model", profile.get("default_model", "dfine_s"))
    student: DFINEAdapter = load_dfine(model_name, device=device)
    student.freeze(profile.get("freeze", []) if exp.get("freeze") is None else exp["freeze"])
    frozen: Optional[DFINEAdapter] = load_dfine(model_name, device=device) if exp.get("use_frozen", True) else None
    if frozen is not None:
        for p in frozen.model.parameters():
            p.requires_grad_(False)
    init_state = {k: v.detach().clone() for k, v in student.model.named_parameters() if v.requires_grad}
    ema: Optional[ModelEMA] = None   # created after the optional AdaBN step so it starts from the same statistics

    crit_cfg = build_criterion_cfg(exp.get("criterion", {}))
    criterion = TCLDCriterion(student, frozen, crit_cfg, input_size, init_state)
    method = exp.get("method", "tcld")
    st_criterion = None
    # "st": self-training / Mean-Teacher only; "mt_tcl": Mean-Teacher loss + hybrid_tcl_weight * TCLD
    # consistency loss on the same clean frame pair (TCLD as a regularizer of EMA self-training).
    hybrid_w = float(exp.get("hybrid_tcl_weight", 1.0)) if method == "mt_tcl" else 0.0
    if method in ("st", "mt_tcl"):
        from tcld.train.self_train import SelfTrainCriterion

        st = exp.get("self_train", {})
        st_criterion = SelfTrainCriterion(student, float(st.get("score_thresh", 0.5)), float(st.get("nms_iou", 0.7)),
                                          bool(st.get("restrict_to_target_classes", False)),
                                          student_aug=st.get("student_aug"), seed=seed)
        st_teacher = st.get("teacher", "frozen")   # frozen | ema
    # "tent" / "actmad": test-time adaptation baselines (tcld.train.tta); both frames of a pair are
    # used as independent unlabeled images, no temporal term, no anchor.
    tta_cfg = exp.get("tta", {})
    actmad = None
    if method == "tent":
        from tcld.train.tta import restrict_to_norm_params

        n_norm = restrict_to_norm_params(student)
        log(f"[{camera}] tent: {n_norm / 1e3:.1f}K norm params trainable")
    elif method == "actmad":
        from tcld.train.tta import ActMADCriterion, source_moments

        cache = Path(os.environ.get("TCLD_RUNS", "runs")) / "cache" / f"actmad_src_{model_name}_{input_size[0]}x{input_size[1]}.pt"
        actmad = ActMADCriterion(student, source_moments(student, input_size, int(tta_cfg.get("source_images", 512)),
                                                         cache=cache))

    # data
    sampler = PairSampler(train_clips, deltas=exp.get("deltas", [1, 2, 5]), base_stride=int(exp.get("base_stride", 30)),
                          max_pairs_per_clip=exp.get("max_pairs_per_clip"), seed=seed)
    ds = PairDataset(sampler, input_size=input_size, photometric=float(exp.get("photometric", 0.0)), seed=seed,
                     flow=(crit_cfg.motion in ("flow", "hybrid")), flow_downscale=int(exp.get("flow_downscale", 2)),
                     independent_views=bool(exp.get("independent_views", False)),
                     n_context=int(crit_cfg.ensemble_k), context_stride=int(exp.get("context_stride", 2)))
    bs = int(exp.get("batch_size", profile.get("batch_size", 2)))
    nw = int(exp.get("num_workers", profile.get("num_workers", 4)))
    # NOTE: no cv2.VideoCapture may be open in this process when workers fork (shared
    # decoder state hangs both sides) -> the probe batch below uses its own dataset.
    loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=nw, collate_fn=collate_pairs, drop_last=True,
                        persistent_workers=False, worker_init_fn=worker_init, timeout=600 if nw else 0,
                        generator=torch.Generator().manual_seed(seed))
    log(f"[{camera}] pairs={len(sampler)} batches/epoch={len(loader)} trainable={sum(p.numel() for p in student.trainable_parameters())/1e6:.2f}M")

    # optim
    lr = float(exp.get("lr", 2e-5))
    opt = torch.optim.AdamW([p for p in student.trainable_parameters()], lr=lr, weight_decay=float(exp.get("weight_decay", 1e-4)))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    iters = int(exp.get("iters", 1000))
    diag_every, eval_every = int(exp.get("diag_every", 50)), int(exp.get("eval_every", 0))
    clip_norm = float(exp.get("clip_max_norm", 0.1))

    # probe batch for diagnostics (fixed)
    ds_probe = PairDataset(sampler, input_size=input_size, seed=seed)
    probe = collate_pairs([ds_probe[i] for i in range(0, len(ds_probe), max(1, len(ds_probe) // 4))][:4])
    ds_probe.close()
    del ds_probe
    probe_imgs = probe["img_a"].to(device)
    with torch.no_grad():
        frozen_probe = frozen.forward_distributions(probe_imgs) if frozen is not None else None

    def run_eval(adp: DFINEAdapter, tag: str) -> Dict[str, float]:
        if not eval_clips:   # unlabeled camera (e.g. Scenes100 frames without manual annotations)
            log(f"[{camera}] eval {tag}: skipped (no GT)")
            return {}
        was_training = adp.model.training
        res = evaluate_cameras(adp, {camera: eval_clips}, device, frame_stride=int(exp.get("eval_frame_stride", 30)),
                               start_frac=eval_start_frac, input_size=input_size,
                               batch_size=int(profile.get("eval_batch_size", 8)), verbose=False)[camera]
        adp.model.train(was_training)
        log(f"[{camera}] eval {tag}: APm_co={res.get('APm_co', float('nan')):.2f} AP50={res.get('AP50_co', float('nan')):.2f} AP75={res.get('AP75_co', float('nan')):.2f}")
        return res

    history: List[dict] = []
    base = run_eval(student, "zero-shot")
    history.append({"iter": 0, "eval_student": base})

    adabn_cfg = exp.get("adabn")
    adabn_eval: Dict[str, float] = {}
    if adabn_cfg:
        # AdaBN: re-estimate BatchNorm statistics of backbone/encoder on target frames (no labels,
        # no gradients); the frozen anchor gets the same statistics so index-aligned queries stay valid.
        from tcld.model import adabn

        n_frames = int(adabn_cfg.get("frames", 256))
        parts = tuple(adabn_cfg.get("parts", ["backbone", "encoder"]))

        def _adabn_batches():
            seen = 0
            for b in loader:
                yield b["img_a"].to(device, non_blocking=True)
                seen += int(b["img_a"].shape[0])
                if seen >= n_frames:
                    break

        prior = float(adabn_cfg.get("prior", 0.5))   # blend with source statistics; 0 (plain AdaBN) collapses the model
        nb = adabn.recalibrate(student, _adabn_batches(), parts=parts, prior=prior)
        if frozen is not None:
            adabn.copy_bn_stats(student, frozen, parts=parts)
            with torch.no_grad():
                frozen_probe = frozen.forward_distributions(probe_imgs)
        log(f"[{camera}] adabn: recalibrated BatchNorm of {parts} on {nb} batches ({n_frames} frames), prior {prior}")
        adabn_eval = run_eval(student, "adabn")
        history.append({"iter": 0, "eval_adabn": adabn_eval})
    if exp.get("use_ema", True):
        ema = ModelEMA(student.model, decay=float(exp.get("ema_decay", 0.999)))

    it, t0 = 0, time.time()
    student.model.train()
    student.set_norm_eval()
    ema_adapter: Optional[DFINEAdapter] = None
    while it < iters:
        for batch in loader:
            if it >= iters:
                break
            img_a, img_b = batch["img_a"].to(device, non_blocking=True), batch["img_b"].to(device, non_blocking=True)
            flow = batch.get("flow")
            teacher_ctx = frozen_ctx = None
            with torch.no_grad():
                fa = frozen.forward_distributions(img_a) if frozen is not None else None
                fb = frozen.forward_distributions(img_b) if frozen is not None else None
                if crit_cfg.ensemble_k > 1 and "img_ctx" in batch and frozen is not None:
                    ctx = batch["img_ctx"].to(device, non_blocking=True)          # [B, k, 3, h, w]
                    frozen_ctx = [frozen.forward_distributions(ctx[:, j]) for j in range(ctx.shape[1] - 1)] + [fa]
                    if exp.get("tcl_teacher", "ema") == "frozen":
                        teacher_ctx = frozen_ctx
                    else:
                        if ema_adapter is None:
                            ema_adapter = student.with_model(ema.module)
                        teacher_ctx = [ema_adapter.forward_distributions(ctx[:, j]) for j in range(ctx.shape[1])]
                teacher_a = None
                if crit_cfg.tcl.mode == "ema":
                    if exp.get("tcl_teacher", "ema") == "frozen" and frozen is not None:
                        teacher_a = fa            # frozen initial model as the target (no co-drift)
                    elif ema is not None:
                        if ema_adapter is None:
                            ema_adapter = student.with_model(ema.module)
                        teacher_a = ema_adapter.forward_distributions(img_a)
            if method in ("tent", "actmad"):
                imgs = torch.cat([img_a, img_b])
                if method == "tent":
                    from tcld.train.tta import tent_loss

                    with torch.autocast("cuda", enabled=amp):
                        out = student.forward_distributions(imgs)
                    loss, stats = tent_loss(out, float(tta_cfg.get("score_min", 0.1)))
                else:
                    loss, stats = actmad(imgs, amp)
                stats.setdefault("loss_tcl", 0.0)
                stats.setdefault("n_gated_out", 0.0)
            elif st_criterion is not None:
                imgs = torch.cat([img_a, img_b])
                with torch.no_grad():
                    if st_teacher == "ema" and ema is not None:
                        if ema_adapter is None:
                            ema_adapter = student.with_model(ema.module)
                        t_out = ema_adapter.forward_distributions(imgs)
                    else:
                        t_out = frozen.forward_distributions(imgs)
                loss, stats = st_criterion(imgs, t_out, amp)
                if method == "mt_tcl":
                    with torch.autocast("cuda", enabled=amp):
                        da = student.forward_distributions(img_a, all_layers=crit_cfg.all_layers)
                        db = student.forward_distributions(img_b, all_layers=crit_cfg.all_layers)
                    with torch.autocast("cuda", enabled=False):
                        loss_t, stats_t = criterion(_to_float(da), _to_float(db), flow, fa, fb, teacher_a, teacher_ctx, frozen_ctx)
                    loss = loss + hybrid_w * loss_t
                    stats_mt = stats
                    stats = dict(stats_t)
                    stats["loss_mt"] = stats_mt["loss_total"]
                    stats.update({k: v for k, v in stats_mt.items() if k.startswith("st_") or k == "n_pseudo"})
                    stats["loss_total"] = float(loss.detach())
                else:
                    stats.setdefault("loss_tcl", 0.0)
                    stats.setdefault("n_matched", stats.get("n_pseudo", 0.0))
                    stats.setdefault("n_gated_out", 0.0)
            else:
                with torch.autocast("cuda", enabled=amp):
                    da = student.forward_distributions(img_a, all_layers=crit_cfg.all_layers)
                    db = student.forward_distributions(img_b, all_layers=crit_cfg.all_layers)
                with torch.autocast("cuda", enabled=False):
                    da_f = _to_float(da)
                    db_f = _to_float(db)
                    loss, stats = criterion(da_f, db_f, flow, fa, fb, teacher_a, teacher_ctx, frozen_ctx)
            opt.zero_grad(set_to_none=True)
            if loss.requires_grad and float(loss) != 0.0:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in student.trainable_parameters()], clip_norm)
                scaler.step(opt)
                scaler.update()
                if ema is not None:
                    ema.update(student.model)
            it += 1
            if it % 10 == 0 or it == 1:
                log(f"[{camera}] it {it}/{iters} loss={stats['loss_total']:.4f} tcl={stats['loss_tcl']:.4f} "
                    f"anchor={stats.get('loss_anchor', 0):.4f} matched={int(stats['n_matched'])} used={int(stats.get('tcl_n_used', 0))} "
                    f"gated={int(stats['n_gated_out'])} ({time.time() - t0:.0f}s)")
            rec = {"iter": it, **stats}
            if diag_every and it % diag_every == 0:
                with torch.no_grad():
                    sp = _to_float(student.forward_distributions(probe_imgs))
                    rec["diag"] = collapse.diagnose(student, sp, frozen_probe, (input_size[1], input_size[0]))
                log(f"[{camera}] diag it {it}: " + ", ".join(f"{k}={v:.3f}" for k, v in rec["diag"].items() if isinstance(v, float)))
            if eval_every and it % eval_every == 0 and it < iters:
                rec["eval_student"] = run_eval(student, f"student it{it}")
            history.append(rec)
            if not torch.isfinite(loss):
                log(f"[{camera}] non-finite loss at it {it}: {stats}")
                break

    final_student = run_eval(student, "student final") if (ema is None or exp.get("eval_student_final", False)) else {"APm_co": -1.0}
    result = {"camera": camera, "seed": seed, "iters": it, "zero_shot": base, "student_final": final_student,
              "runtime_s": time.time() - t0}
    if adabn_cfg:
        result["adabn"] = adabn_eval
    if ema is not None:
        ema_adapter = student.with_model(ema.module)
        result["ema_final"] = run_eval(ema_adapter, "ema final")
    best_key = "student_final"
    if "ema_final" in result and (not eval_clips or
                                  result["ema_final"].get("APm_co", -1) > final_student.get("APm_co", -1)):
        best_key = "ema_final"
    result["final"] = result[best_key]
    result["final_source"] = best_key
    ckpt = out_dir / f"adapt_{camera}_seed{seed}.pth"
    if os.environ.get("TCLD_NO_CKPT", "0") != "1":  # disk saver for replicate seeds
        torch.save({"model": (ema.module if best_key == "ema_final" else student.model).state_dict(),
                    "camera": camera, "seed": seed, "exp": exp}, ckpt)
        result["checkpoint"] = str(ckpt)
    (out_dir / f"history_{camera}_seed{seed}.json").write_text(json.dumps(history, default=float))
    (out_dir / f"result_{camera}_seed{seed}.json").write_text(json.dumps(result, default=float, indent=1))
    return result


def _to_float(d):
    """AMP forward returns fp16 tensors; losses on distributions must be fp32."""
    from dataclasses import replace

    return replace(d, logits=d.logits.float(), boxes=d.boxes.float(), corners=d.corners.float(),
                   ref_points=d.ref_points.float(), aux=[_to_float(x) for x in d.aux])
