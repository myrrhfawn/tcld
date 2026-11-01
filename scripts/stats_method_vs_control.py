"""Camera-level 'method − control' tables for the paper-extension arms (supervisor tasks 1 and 2, 2026-10-26).

Unlike stats_baselines.py (baseline families against the method on MOT, D-FINE-S), this script compares ONE
method arm against ONE control arm on any dataset / detector, pairing on the seeds present in both arms:

  --preset virat    final recipe on VIRAT, 11 scenes, eval every 30th frame of the held-out clips
  --preset L        D-FINE-L (batch 1, N0 16, 1500 iters) on clean / night / jpeg / xrec, printed next to the
                    corresponding D-FINE-S rows (same cameras, same seeds where available)
  --preset xrec5    cross-recording protocol with the 5th pair MOT17-02 <- MOT17-01 (method and control, frozen + enc)
  --method EXP --control EXP [--dataset mot|virat] [--cams ...]   any custom pair

Per camera: zero-shot APm, Δ over zero-shot of both arms (mean over common seeds), method − control; then
mean, sign count and the exact two-sided Wilcoxon p with its floor 2 / 2^n. Camera is the unit of analysis.
"""
import argparse, json, math, os
import numpy as np
from scipy import stats

ROOT = os.environ.get("TCLD_RUNS", "runs")
SEEDS = [0, 1, 2]
METRICS = ("APm_co", "AP50_co", "AP75_co")
METHOD_S, CTRL_S = "tf_a03_w5_n32_iou08_dense", "ctrl_a03_dense"
METHOD_L, CTRL_L = "tf_a03_w5_n16_iou08_dense", "ctrl_a03_dense"
C6 = ["MOT17-02", "MOT17-04", "MOT17-09", "MOT20-01", "MOT20-03", "MOT20-05"]
C7 = ["MOT17-02", "MOT17-04", "MOT17-09", "MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]
X5 = ["MOT17-02", "MOT17-04", "MOT20-01", "MOT20-03", "MOT20-05"]
V11 = ["0000", "0001", "0002", "0100", "0101", "0102", "0400", "0401", "0500", "0502", "0503"]
SUF = {"clean": "", "night": "_night", "jpeg": "_jpeg", "xrec": "_xrec"}
CAMS = {"clean": C6, "night": C7, "jpeg": C7, "xrec": X5}


def load(exp, ds, cam):
    """{seed: {'zs': {metric}, 'delta': {metric}}} for the seeds whose result file exists."""
    out = {}
    for s in SEEDS:
        p = f"{ROOT}/{exp}/{ds}/result_{cam}_seed{s}.json"
        if os.path.exists(p):
            d = json.load(open(p))
            if "final" in d and "zero_shot" in d:
                out[s] = {"zs": {m: d["zero_shot"][m] for m in METRICS},
                          "delta": {m: d["final"][m] - d["zero_shot"][m] for m in METRICS}}
    return out


def wilcoxon(x):
    x = np.asarray([v for v in x if not math.isnan(v)], float)
    n = len(x)
    res = {"n": n, "mean": float(x.mean()) if n else float("nan"), "pos": int((x > 0).sum()),
           "floor": 2.0 / 2 ** n if n else float("nan"), "p": float("nan")}
    try:
        if n >= 2 and np.any(x != 0):
            res["p"] = float(stats.wilcoxon(x, alternative="two-sided", method="exact").pvalue)
    except Exception:
        pass
    return res


def row(label, per_cam, cams):
    t = wilcoxon(per_cam.values())
    cells = " ".join(f"{per_cam[c]:+6.2f}" if not math.isnan(per_cam[c]) else "   n/a" for c in cams)
    return f"  {label:30s} {cells} | mean {t['mean']:+.2f} pos {t['pos']}/{t['n']} Wilcoxon p={t['p']:.3f} (floor {t['floor']:.3f})"


def block(title, method, control, ds, cams):
    M = {c: load(method, ds, c) for c in cams}
    C = {c: load(control, ds, c) for c in cams}
    common = {c: sorted(set(M[c]) & set(C[c])) for c in cams}
    if not any(common.values()):
        return [f"\n=== {title}: no common seeds yet ({method} vs {control})"]
    L = [f"\n=== {title}  [{method} vs {control}, {ds}] common seeds per camera: "
         + " ".join(f"{c}:{''.join(map(str, common[c])) or '-'}" for c in cams)]
    L.append("  " + " " * 30 + " ".join(f"{c[-5:]:>6s}" for c in cams))
    zs = {c: float(np.mean([M[c][s]["zs"]["APm_co"] for s in common[c]])) if common[c] else float("nan") for c in cams}
    L.append("  " + f"{'APm_co zero-shot':30s} " + " ".join(f"{zs[c]:6.2f}" if not math.isnan(zs[c]) else "   n/a" for c in cams))
    for m in METRICS:
        dm = {c: float(np.mean([M[c][s]["delta"][m] for s in common[c]])) if common[c] else float("nan") for c in cams}
        dc = {c: float(np.mean([C[c][s]["delta"][m] for s in common[c]])) if common[c] else float("nan") for c in cams}
        L.append(row(f"{m} method Δ over zero-shot", dm, cams))
        L.append(row(f"{m} control Δ over zero-shot", dc, cams))
        L.append(row(f"{m} method − control", {c: dm[c] - dc[c] for c in cams}, cams))
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["virat", "L", "xrec5"])
    ap.add_argument("--method"); ap.add_argument("--control")
    ap.add_argument("--dataset", default="mot"); ap.add_argument("--cams", nargs="*")
    ap.add_argument("--out")
    a = ap.parse_args()
    lines = []
    if a.preset == "virat":
        lines += block("VIRAT, 11 scenes, final recipe (S)", METHOD_S + "_virat", CTRL_S + "_virat", "virat", V11)
    elif a.preset == "L":
        for cond in ("clean", "night", "jpeg", "xrec"):
            lines += block(f"D-FINE-L {cond}", METHOD_L + SUF[cond] + "_L", CTRL_L + SUF[cond] + "_L", "mot", CAMS[cond])
            lines += block(f"D-FINE-S {cond} (reference rows)", METHOD_S + SUF[cond], CTRL_S + SUF[cond], "mot", CAMS[cond])
    elif a.preset == "xrec5":
        for enc in ("", "_enc"):
            lines += block(f"cross-recording, 5 pairs{' [enc]' if enc else ' [frozen]'}",
                           METHOD_S + enc + "_xrec", CTRL_S + enc + "_xrec", "mot", X5)
    else:
        if not (a.method and a.control):
            ap.error("--preset or --method/--control")
        lines += block(f"{a.method} vs {a.control}", a.method, a.control, a.dataset, a.cams or C6)
    text = "\n".join(lines).lstrip("\n") + "\n"
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        open(a.out, "w").write(text)


if __name__ == "__main__":
    main()
