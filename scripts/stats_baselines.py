"""Camera-level comparison of baselines against TCL and the control in the final protocol.

For every baseline family (st, mt, sameframe, moments, coord, mttcl_w1, mttcl_w10, mtanchor, tent, actmad) and arm (frozen / enc) this prints,
per condition (clean n=6, night n=7, jpeg n=7, shift pooled n=7), the per-camera Δ over
zero-shot (mean over the seeds present), sign counts and the exact Wilcoxon p (with its
floor for the given n) of baseline − TCL and baseline − ctrl. Camera is the unit of analysis;
cells still missing are skipped per camera (comparisons use the seeds present in both arms).

Cross-recording mode (--conds xrec): adapt on the MOT test recording, evaluate on the train one;
the eval cameras are the unit (5 pairs since 2026-10-26: MOT17-02<-01, MOT17-04<-03, MOT20-01<-07,
MOT20-03<-04, MOT20-05<-04; n=5, floor 0.062; families run on the first 4 pairs only show n/a for MOT17-02),
runs are named <base>[_enc]_xrec.

Usage: python scripts/stats_baselines.py [--out results/tables/stats_baselines.txt] [--conds xrec]
"""
import argparse, json, math, os
import numpy as np
from scipy import stats

ROOT = os.environ.get("TCLD_RUNS", "runs")
SEEDS = [0, 1, 2]
CAMS6 = ["MOT17-02", "MOT17-04", "MOT17-09", "MOT20-01", "MOT20-03", "MOT20-05"]
CAMS7 = ["MOT17-02", "MOT17-04", "MOT17-09", "MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]
CAMS5 = ["MOT17-02", "MOT17-04", "MOT20-01", "MOT20-03", "MOT20-05"]   # cross-recording eval cameras
CAMS = {"clean": CAMS6, "night": CAMS7, "jpeg": CAMS7, "xrec": CAMS5}
METRICS = ("APm_co", "AP50_co", "AP75_co")
FAMILIES = {
    "st": "st_a03_dense",
    "mt": "mt_a03_dense",
    "sameframe": "sameframe_a03_w5_n32_iou08_dense",
    "moments": "moments_a03_w5_n32_iou08_dense",
    "coord": "coord_a03_w5_n32_iou08_dense",
    "mttcl_w1": "mttcl_w1_a03_dense",
    "mttcl_w10": "mttcl_w10_a03_dense",
    "mtanchor": "mtanchor_a03_dense",
    "mtdose_w01": "mtdose_w01_a03_dense",
    "mtdose_w05": "mtdose_w05_a03_dense",
    "mtdose_w2": "mtdose_w2_a03_dense",
    "tent": "tent_a03_dense",
    "actmad": "actmad_a03_dense",
}
REF = {"tcl": "tf_a03_w5_n32_iou08_dense", "ctrl": "ctrl_a03_dense", "mt": "mt_a03_dense",
       "mttcl_w10": "mttcl_w10_a03_dense"}
# references each family is compared against: the MT+TCL hybrids also against plain MT; the anchor-only
# attribution arm (mtanchor) also against MT and against the full hybrid mttcl_w10
def REFS_FOR(fam):
    if fam == "mtanchor":
        return ("tcl", "ctrl", "mt", "mttcl_w10")
    return ("tcl", "ctrl", "mt") if fam.startswith("mttcl") else ("tcl", "ctrl")
SUFFIX = {"clean": "", "night": "_night", "jpeg": "_jpeg"}


def exp_name(base, cond, enc):
    if cond == "xrec":
        return base + ("_enc" if enc else "") + "_xrec"
    return base + SUFFIX[cond] + ("_enc" if enc else "")


def load(exp, cam):
    """{seed: {metric: Δ over zero-shot}} for the seeds whose result file exists."""
    out = {}
    for s in SEEDS:
        p = f"{ROOT}/{exp}/mot/result_{cam}_seed{s}.json"
        if os.path.exists(p):
            d = json.load(open(p))
            if "final" not in d or "zero_shot" not in d:
                continue
            out[s] = {m: d["final"][m] - d["zero_shot"][m] for m in METRICS}
    return out


def clean_ctrl_frozen(cam):
    """Clean frozen control. Since the protocol fix of 2026-10-25 (eval stride 30 -> 3) the redone run
    runs/ctrl_a03_dense/mot/result_<cam>_seed<k>.json is used directly; while a camera is not redone yet,
    fall back to the dense (stride 3) re-eval of the old checkpoints in runs/eval/dense3_ctrl_a03_dense_<cam>_s<k>.json."""
    out = load(exp_name(REF["ctrl"], "clean", False), cam)
    if out:
        return out
    zs = None
    for f in ("dense3_zeroshot.json", "dense3_zeroshot_heldout.json"):
        p = f"{ROOT}/eval/{f}"
        if os.path.exists(p):
            d = json.load(open(p))
            if cam in d["per_camera"]:
                zs = d["per_camera"][cam]
                break
    if zs is None:
        return out
    for s in SEEDS:
        p = f"{ROOT}/eval/dense3_ctrl_a03_dense_{cam}_s{s}.json"
        if os.path.exists(p):
            fin = json.load(open(p))["per_camera"][cam]
            out[s] = {m: fin[m] - zs[m] for m in METRICS}
    return out


def wilcoxon(x):
    x = np.asarray(x, float)
    n = len(x)
    res = {"n": n, "mean": float(x.mean()) if n else float("nan"),
           "pos": int((x > 0).sum()), "floor": 2.0 / 2 ** n if n else float("nan")}
    try:
        res["p"] = float(stats.wilcoxon(x, alternative="two-sided", method="exact").pvalue) if n >= 2 else float("nan")
    except Exception:
        res["p"] = float("nan")
    return res


def cam_delta(table, cam, metric, seeds=None):
    d = table.get(cam, {})
    ss = [s for s in d if seeds is None or s in seeds]
    return float(np.mean([d[s][metric] for s in ss])) if ss else float("nan")


def compare(a, b, cams, metric):
    """Per-camera (a − b) using the seeds present in both; cameras missing in either arm are NaN."""
    out = {}
    for c in cams:
        common = set(a.get(c, {})) & set(b.get(c, {}))
        out[c] = cam_delta(a, c, metric, common) - cam_delta(b, c, metric, common) if common else float("nan")
    return out


def fmt_row(label, per_cam, cams):
    x = np.array([v for v in per_cam.values() if not math.isnan(v)])
    t = wilcoxon(x)
    cells = " ".join(f"{per_cam[c]:+6.2f}" if not math.isnan(per_cam[c]) else "   n/a" for c in cams)
    return (f"  {label:28s} {cells} | mean {t['mean']:+.2f} pos {t['pos']}/{t['n']} "
            f"Wilcoxon p={t['p']:.3f} (floor {t['floor']:.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/tables/stats_baselines.txt")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--conds", default="clean,night,jpeg")
    args = ap.parse_args()
    lines = []
    P = lines.append
    for fam in args.families.split(","):
        base = FAMILIES[fam]
        for enc in (False, True):
            arm = "enc" if enc else "frozen"
            pooled = {}  # (metric, ref) -> {cam: [per-cond values]}
            for cond in args.conds.split(","):
                cams = CAMS[cond]
                bl = {c: load(exp_name(base, cond, enc), c) for c in cams}
                seeds_present = {c: sorted(bl[c]) for c in cams}
                if not any(seeds_present.values()):
                    continue
                tcl = {c: load(exp_name(REF["tcl"], cond, enc), c) for c in cams}
                if cond == "clean" and not enc:
                    ctrl = {c: clean_ctrl_frozen(c) for c in cams}
                else:
                    ctrl = {c: load(exp_name(REF["ctrl"], cond, enc), c) for c in cams}
                refs = {"ctrl": ctrl, "tcl": tcl}
                for r in REFS_FOR(fam):
                    if r not in refs:
                        refs[r] = {c: load(exp_name(REF[r], cond, enc), c) for c in cams}
                P(f"\n=== {fam} [{arm}] {cond} (n={len(cams)}) seeds per camera: "
                  + " ".join(f"{c}:{''.join(map(str, seeds_present[c]))}" for c in cams))
                P("  " + " " * 28 + " ".join(f"{c[-5:]:>6s}" for c in cams))
                for metric in METRICS:
                    for name, tab in list(refs.items()) + [(fam, bl)]:
                        pc = {c: cam_delta(tab, c, metric) for c in cams}
                        P(fmt_row(f"{metric} {name} Δ over zero-shot", pc, cams))
                    for name in REFS_FOR(fam):
                        pc = compare(bl, refs[name], cams, metric)
                        P(fmt_row(f"{metric} {fam} − {name}", pc, cams))
                        if cond in ("night", "jpeg"):
                            for c in cams:
                                pooled.setdefault((metric, name), {}).setdefault(c, []).append(pc[c])
            if pooled:
                P(f"\n=== {fam} [{arm}] shift pooled (night+jpeg averaged per camera, n=7)")
                for metric in METRICS:
                    for name in REFS_FOR(fam):
                        pc = {c: float(np.nanmean(v)) if not all(math.isnan(x) for x in v) else float("nan")
                              for c, v in pooled[(metric, name)].items()}
                        P(fmt_row(f"{metric} {fam} − {name}", pc, CAMS7))
    text = "\n".join(lines).lstrip("\n") + "\n"
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        open(args.out, "w").write(text)


if __name__ == "__main__":
    main()
