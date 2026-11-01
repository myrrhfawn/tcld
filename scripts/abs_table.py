"""Absolute AP table: mAP@[.5:.95], AP50 and AP75 of every compared method per evaluation setting.

Reads runs/<exp>/<dataset>/result_<camera>_seed<s>.json (written by scripts/run_adapt.py), averages the
final model's metrics over seeds within a camera and then over cameras (camera = unit of analysis), and
prints one table per setting with the zero-shot (unadapted) detector as the first row. Within a table every
row is computed on the same cameras (the intersection of cameras available to all rows), so the columns are
directly comparable; rows missing a setting are omitted. Zero-shot numbers come from the method's result files
(the audit in scripts/audit_protocol.py checks that every arm shares them).

Usage: python scripts/abs_table.py [--out results/tables/absolute_ap.md] [--variants frozen,enc,L]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
from typing import Dict, List, Optional, Tuple

RUNS = os.environ.get("TCLD_RUNS", "runs")
METRICS = ("APm_co", "AP50_co", "AP75_co")

C6 = ["MOT17-02", "MOT17-04", "MOT17-09", "MOT20-01", "MOT20-03", "MOT20-05"]
C7 = ["MOT17-02", "MOT17-04", "MOT17-09", "MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]
X5 = ["MOT17-04", "MOT20-01", "MOT20-03", "MOT20-05", "MOT17-02"]
V11 = ["0000", "0001", "0002", "0100", "0101", "0102", "0400", "0401", "0500", "0502", "0503"]

# setting -> (dataset, cameras, suffix appended to the family name)
SETTINGS = {
    "clean": ("mot", C6, ""),
    "night": ("mot", C7, "_night"),
    "jpeg": ("mot", C7, "_jpeg"),
    "xrec": ("mot", X5, "_xrec"),
    "virat": ("virat", V11, "_virat"),
}
SETTING_TITLE = {
    "clean": "MOT, original recordings (adapt on the first 70 % of frames, evaluate on the rest)",
    "night": "MOT, synthetic night shift (gamma darkening, sensor noise, JPEG q=30)",
    "jpeg": "MOT, synthetic compression shift (1/3 scale, JPEG q=15, upscaled)",
    "xrec": "MOT, cross-recording (adapt on the unlabeled test recording, evaluate on the labeled one)",
    "virat": "VIRAT Ground 2.0, 11 scenes (adapt on 70 % of clips, evaluate every 30th frame of the rest)",
}

# display name -> family prefix (D-FINE-S). The setting suffix and, for the encoder variant, "_enc" are appended.
FAMILIES_S = [
    ("TCLD (ours)", "tf_a03_w5_n32_iou08_dense"),
    ("control (same recipe, no TCLD term)", "ctrl_a03_dense"),
    ("Tent", "tent_a03_dense"),
    ("ActMAD", "actmad_a03_dense"),
    ("self-training (hard pseudo-boxes)", "st_a03_dense"),
    ("Mean Teacher", "mt_a03_dense"),
    ("Mean Teacher + TCLD", "mttcl_w10_a03_dense"),
    ("coordinate consistency (L1 + GIoU)", "coord_a03_w5_n32_iou08_dense"),
    ("moment consistency (edge mean, std)", "moments_a03_w5_n32_iou08_dense"),
    ("same-frame two-view W1", "sameframe_a03_w5_n32_iou08_dense"),
]
FAMILIES_L = [
    ("TCLD (ours)", "tf_a03_w5_n16_iou08_dense"),
    ("control (same recipe, no TCLD term)", "ctrl_a03_dense"),
]


def exp_name(prefix: str, setting: str, variant: str) -> str:
    suffix = SETTINGS[setting][2]
    if variant == "frozen":
        return prefix + suffix
    if variant == "enc":
        # historical naming: *_enc_xrec for the cross-recording protocol, *_<shift>_enc otherwise
        return prefix + ("_enc_xrec" if setting == "xrec" else suffix + "_enc")
    if variant == "L":
        return prefix + suffix + "_L"
    raise ValueError(variant)


def load(exp: str, dataset: str, seeds: Optional[set] = None) -> Dict[str, Dict[str, Dict[str, float]]]:
    """camera -> {'zero_shot': metrics, 'final': metrics} averaged over seeds; plus the seed list."""
    out: Dict[str, Dict] = {}
    per_cam: Dict[str, List[dict]] = {}
    for f in glob.glob(f"{RUNS}/{exp}/{dataset}/result_*_seed*.json"):
        m = re.search(r"result_(.+)_seed(\d+)\.json$", f)
        if not m or (seeds is not None and int(m.group(2)) not in seeds):
            continue
        d = json.load(open(f))
        d["_seed"] = int(m.group(2))
        per_cam.setdefault(m.group(1), []).append(d)
    for cam, ds in per_cam.items():
        out[cam] = {
            "zero_shot": {k: statistics.mean(x["zero_shot"][k] for x in ds) for k in METRICS},
            "final": {k: statistics.mean(x["final"][k] for x in ds) for k in METRICS},
            "seeds": sorted(x["_seed"] for x in ds),
        }
    return out


def mean_over(cams: List[str], data: Dict, key: str) -> Tuple[float, ...]:
    return tuple(statistics.mean(data[c][key][k] for c in cams) for k in METRICS)


def fmt(vals: Tuple[float, ...]) -> str:
    return " | ".join(f"{v:.2f}" for v in vals)


# the D-FINE-L protocol was run on seed 0 only; a stray extra cell must not enter its means
VARIANT_SEEDS = {"frozen": None, "enc": None, "L": {0}}
JSON: Dict[str, Dict] = {}


def table(setting: str, variant: str, families) -> Optional[List[str]]:
    dataset, cams, _ = SETTINGS[setting]
    rows = []
    for name, prefix in families:
        data = load(exp_name(prefix, setting, variant), dataset, VARIANT_SEEDS[variant])
        if data:
            rows.append((name, data))
    if not rows:
        return None
    common = [c for c in cams if all(c in d for _, d in rows)]
    if not common:
        return None
    seeds = sorted({s for _, d in rows for c in common for s in d[c]["seeds"]})
    lines = [
        f"### {SETTING_TITLE[setting]}",
        "",
        f"Model: {'D-FINE-L' if variant == 'L' else 'D-FINE-S'}; "
        f"{'encoder trainable, backbone frozen' if variant == 'enc' else 'backbone and encoder frozen, decoder trainable'}; "
        f"n = {len(common)} cameras ({', '.join(common)}); seeds {''.join(map(str, seeds))} "
        f"(mean over seeds, then over cameras).",
        "",
        "| method | mAP | AP50 | AP75 |",
        "|---|---|---|---|",
        f"| zero-shot (no adaptation) | {fmt(mean_over(common, rows[0][1], 'zero_shot'))} |",
    ]
    rec = {"cams": common, "seeds": "".join(map(str, seeds)), "rows": [["zero-shot", "", *mean_over(common, rows[0][1], "zero_shot")]]}
    for name, data in rows:
        per_row_seeds = sorted({s for c in common for s in data[c]["seeds"]})
        tag = "" if per_row_seeds == seeds else f" (seeds {''.join(map(str, per_row_seeds))})"
        lines.append(f"| {name}{tag} | {fmt(mean_over(common, data, 'final'))} |")
        rec["rows"].append([name, tag.strip(), *mean_over(common, data, "final")])
    JSON[f"{variant}/{setting}"] = rec
    skipped = [c for c in cams if c not in common]
    if skipped:
        lines += ["", f"Cameras without results for every row and therefore excluded here: {', '.join(skipped)}."]
        # the method / control pair is usually complete: report it on the full set as well
        full = [(n, d) for n, d in rows[:2] if all(c in d for c in cams)]
        if len(full) == 2:
            lines.append(
                f"On all {len(cams)} cameras: zero-shot {fmt(mean_over(cams, full[0][1], 'zero_shot'))}; "
                f"{full[0][0]} {fmt(mean_over(cams, full[0][1], 'final'))}; {full[1][0]} {fmt(mean_over(cams, full[1][1], 'final'))} (mAP | AP50 | AP75)."
            )
    lines.append("")
    return lines


def per_camera(variant: str = "frozen") -> List[str]:
    lines = ["### Per-camera mAP (D-FINE-S, frozen encoder): zero-shot / TCLD / control", "",
             "| setting | camera | zero-shot | TCLD | control | seeds |", "|---|---|---|---|---|---|"]
    for setting, (dataset, cams, _) in SETTINGS.items():
        m = load(exp_name(FAMILIES_S[0][1], setting, variant), dataset)
        c = load(exp_name(FAMILIES_S[1][1], setting, variant), dataset)
        for cam in cams:
            if cam in m and cam in c:
                lines.append(f"| {setting} | {cam} | {m[cam]['zero_shot']['APm_co']:.2f} | {m[cam]['final']['APm_co']:.2f} | "
                             f"{c[cam]['final']['APm_co']:.2f} | {''.join(map(str, sorted(set(m[cam]['seeds']) & set(c[cam]['seeds']))))} |")
    lines.append("")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/tables/absolute_ap.md")
    ap.add_argument("--variants", default="frozen,enc,L")
    ap.add_argument("--json", default="results/tables/absolute_ap.json", help="machine-readable copy of the tables")
    a = ap.parse_args()
    out: List[str] = ["# Absolute AP of every compared method", "",
                      "mAP = COCO mAP@[.5:.95], AP50, AP75 over the two target classes (person, vehicle), "
                      "Scenes100-style per-camera evaluation with ignore regions. Generated by scripts/abs_table.py.", ""]
    heads = {"frozen": "## D-FINE-S, frozen backbone and encoder (main setting)",
             "enc": "## D-FINE-S, trainable encoder",
             "L": "## D-FINE-L (batch 1, N0 16, 1500 iterations; seed 0)"}
    for variant in a.variants.split(","):
        out += [heads[variant], ""]
        for setting in SETTINGS:
            t = table(setting, variant, FAMILIES_L if variant == "L" else FAMILIES_S)
            if t:
                out += t
    out += per_camera()
    text = "\n".join(out)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        f.write(text)
    with open(a.json, "w") as f:
        json.dump(JSON, f, indent=1)
    print(text)


if __name__ == "__main__":
    main()
