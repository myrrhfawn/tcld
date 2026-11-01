"""Protocol audit for the final experiment families (run before any queue launch, FINDINGS entry or report table).

1. One-factor check: every comparison arm must differ from its reference config only in the keys that define it.
2. Variant check: *_enc / *_night / *_jpeg / *_xrec configs differ from the base only in freeze / corruption / eval keys;
   the VIRAT arm differs only in eval stride; the D-FINE-L arms only in model / batch / N0.
3. Result check: within one (dataset, condition, model) every family has the same zero-shot APm per camera
   (same eval frames), all cells have the expected number of iterations (S 750, L 1500), and comparisons will be
   paired on common seeds.
Exit code 1 if anything is flagged.
"""
import glob, json, os, re, sys, collections
import yaml

METHOD = "tf_a03_w5_n32_iou08_dense"
METHOD_L = "tf_a03_w5_n16_iou08_dense"
# reference -> arm : keys that are ALLOWED to differ (anything else is flagged)
PAIRS = {
    (METHOD, "ctrl_a03_dense"): {"criterion.tcl_weight"},
    (METHOD, "coord_a03_w5_n32_iou08_dense"): {"method", "criterion.method"},
    (METHOD, "moments_a03_w5_n32_iou08_dense"): {"method", "criterion.method"},
    (METHOD, "sameframe_a03_w5_n32_iou08_dense"): {"criterion.motion", "criterion.min_motion_px", "deltas", "independent_views", "photometric"},
    ("mt_a03_dense", "st_a03_dense"): {"self_train.teacher", "self_train.student_aug.hflip", "self_train.student_aug.photometric"},
    ("mttcl_w10_a03_dense", "mtanchor_a03_dense"): {"criterion.tcl_weight"},
    ("mttcl_w10_a03_dense", "mttcl_w1_a03_dense"): {"hybrid_tcl_weight"},
    ("mttcl_w10_a03_dense", "mtdose_w01_a03_dense"): {"criterion.tcl_weight"},
    ("mttcl_w10_a03_dense", "mtdose_w05_a03_dense"): {"criterion.tcl_weight"},
    ("mttcl_w10_a03_dense", "mtdose_w2_a03_dense"): {"criterion.tcl_weight"},
    (METHOD + "_virat", "ctrl_a03_dense_virat"): {"criterion.tcl_weight"},
    (METHOD_L + "_L", "ctrl_a03_dense_L"): {"criterion.tcl_weight"},
    (METHOD_L + "_night_L", "ctrl_a03_dense_night_L"): {"criterion.tcl_weight"},
    (METHOD_L + "_jpeg_L", "ctrl_a03_dense_jpeg_L"): {"criterion.tcl_weight"},
    (METHOD_L + "_xrec_L", "ctrl_a03_dense_xrec_L"): {"criterion.tcl_weight"},
}
# arms whose loss family differs entirely from the method: only the shared training schedule must match
SCHEDULE_KEYS = {"iters", "lr", "weight_decay", "clip_max_norm", "batch_size", "use_frozen", "use_ema", "ema_decay",
                 "eval_every", "eval_frame_stride", "deltas", "base_stride", "max_pairs_per_clip", "model", "freeze", "corruption"}
# declared exception: lr chosen by the labelled sweep in the baseline's favour (FINDINGS 2026-10-24)
LR_SWEPT = {"tent_a03_dense", "actmad_a03_dense"}
SCHEDULE_PAIRS = [(METHOD, x) for x in ("mt_a03_dense", "st_a03_dense", "tent_a03_dense", "actmad_a03_dense", "mttcl_w10_a03_dense")]
SUFFIXES = ["", "_enc", "_night", "_jpeg", "_night_enc", "_jpeg_enc", "_xrec", "_enc_xrec"]
VARIANT_KEYS = {"name", "freeze", "corruption", "eval_every", "eval_frame_stride"}
# base -> derived arm : allowed differences (dataset / detector arms of the method, FINDINGS 2026-10-26)
DERIVED = {
    (METHOD, METHOD + "_virat"): {"eval_frame_stride"},
    **{(METHOD + s, METHOD_L + s + "_L"): {"model", "batch_size", "criterion.tcl.min_denom_pairs"} for s in ("", "_night", "_jpeg", "_xrec")},
}
FAMILIES = [METHOD, "ctrl_a03_dense", "coord_a03_w5_n32_iou08_dense", "moments_a03_w5_n32_iou08_dense", "sameframe_a03_w5_n32_iou08_dense",
            "mt_a03_dense", "st_a03_dense", "mttcl_w10_a03_dense", "mttcl_w1_a03_dense", "mtanchor_a03_dense",
            "mtdose_w01_a03_dense", "mtdose_w05_a03_dense", "mtdose_w2_a03_dense", "tent_a03_dense", "actmad_a03_dense"]
EXPS = [f + s for f in FAMILIES for s in SUFFIXES]
EXPS += [METHOD + "_virat", "ctrl_a03_dense_virat"]
EXPS += [m + s + "_L" for m in (METHOD_L, "ctrl_a03_dense") for s in ("", "_night", "_jpeg", "_xrec")]
EXPS = [e.replace("tf_a03_w5_n16_iou08_dense", METHOD_L) for e in EXPS]


def flat(d, p=""):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            out.update(flat(v, p + k + "."))
        else:
            out[p + k] = v
    return out


def cfg(name):
    p = f"configs/exp/{name}.yml"
    return flat(yaml.safe_load(open(p))) if os.path.exists(p) else None


def diff(a, b):
    A, B = cfg(a), cfg(b)
    if A is None or B is None:
        return None
    return {k: (A.get(k, "<absent>"), B.get(k, "<absent>")) for k in sorted(set(A) | set(B))
            if A.get(k, "<absent>") != B.get(k, "<absent>") and k != "name"}


flags = []
for suf in SUFFIXES:
    for (ref, arm), allowed in PAIRS.items():
        d = diff(ref + suf, arm + suf)
        if d is None:
            continue
        bad = {k: v for k, v in d.items() if k not in allowed}
        if bad:
            flags.append(f"[one-factor] {arm}{suf} vs {ref}{suf}: unexpected {bad}")
    for ref, arm in SCHEDULE_PAIRS:
        d = diff(ref + suf, arm + suf)
        if d is None:
            continue
        bad = {k: v for k, v in d.items() if k in SCHEDULE_KEYS and not (k == "lr" and arm in LR_SWEPT)}
        if bad:
            flags.append(f"[schedule] {arm}{suf} vs {ref}{suf}: {bad}")
for base in FAMILIES:
    for a, b in [("", "_enc"), ("", "_night"), ("", "_jpeg"), ("", "_xrec"), ("_night", "_night_enc"), ("_jpeg", "_jpeg_enc"), ("_xrec", "_enc_xrec")]:
        d = diff(base + a, base + b)
        if d is None:
            continue
        bad = {k: v for k, v in d.items() if k not in VARIANT_KEYS}
        if bad:
            flags.append(f"[variant] {base}{a} vs {base}{b}: {bad}")
for (base, arm), allowed in DERIVED.items():
    d = diff(base, arm)
    if d is None:
        flags.append(f"[derived] missing config for {base} -> {arm}")
        continue
    bad = {k: v for k, v in d.items() if k not in allowed}
    if bad:
        flags.append(f"[derived] {arm} vs {base}: unexpected {bad}")

zs = collections.defaultdict(lambda: collections.defaultdict(set))
iters = collections.defaultdict(set)
for exp in EXPS:
    model = "L" if exp.endswith("_L") else "S"
    cond = ("xrec" if "_xrec" in exp else "night" if "_night" in exp else "jpeg" if "_jpeg" in exp else "clean")
    for ds in ("mot", "virat"):
        for f in glob.glob(f"runs/{exp}/{ds}/result_*_seed*.json"):
            cam = re.search(r"result_(.+)_seed", f).group(1)
            j = json.load(open(f))
            zs[(ds, cond, model, cam)][round(j["zero_shot"]["APm_co"], 2)].add(exp)
            iters[exp].add(j.get("iters"))
for key, groups in sorted(zs.items()):
    if len(groups) > 1:
        flags.append(f"[zero-shot] {key}: " + " | ".join(f"{z}: {sorted(e)}" for z, e in groups.items()))
for exp, its in sorted(iters.items()):
    want = 1500 if exp.endswith("_L") else 750
    if len(its) > 1 or (its and its != {want}):
        flags.append(f"[iters] {exp}: {sorted(its)} (expected {want})")

if flags:
    print("\n".join(flags))
    print(f"\n{len(flags)} flag(s)")
    sys.exit(1)
print("protocol audit: OK (one-factor diffs, variants, derived arms, zero-shot per dataset/condition/model, iters S 750 / L 1500)")
