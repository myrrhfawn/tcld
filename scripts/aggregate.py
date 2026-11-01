"""Rebuild results/tables/*.md from results/experiments.csv (P0.6 / P6.4).

Tables
  all_experiments.md  — every row, compact;
  by_method.md        — per experiment config: cameras covered, mean base/final AP,
                        mean APG (m / 50 / 75) ± std over cameras & seeds;
  per_camera.md       — exp × camera matrix of APG_m_co (mean over seeds).
Base AP is reconstructed as final − APG.
"""

from __future__ import annotations

import pandas as pd

from tcld.paths import RESULTS


def _md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".2f")


def main() -> None:
    csv = RESULTS / "experiments.csv"
    df = pd.read_csv(csv)
    out_dir = RESULTS / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        print(f"{csv}: no experiments yet")
        return
    df["exp"] = df["config_path"].str.replace(r".*/", "", regex=True).str.replace(".yml", "", regex=False)
    df["base_APm"] = df["APm_co"] - df["APG_m_co"]
    df["base_AP50"] = df["AP50_co"] - df["APG_50_co"]

    cols = ["exp_id", "method", "model", "cameras", "seed", "iters", "motion", "loss_variant", "reg",
            "base_APm", "APm_co", "APG_m_co", "APG_50_co", "APG_75_co", "runtime_min", "notes"]
    (out_dir / "all_experiments.md").write_text("# All experiments\n\n" + _md(df[[c for c in cols if c in df.columns]]) + "\n")

    g = df.groupby(["exp", "method", "motion", "loss_variant", "reg"], dropna=False)
    rows = []
    for key, grp in g:
        rows.append({
            "exp": key[0], "method": key[1], "motion": key[2], "loss": key[3], "reg": key[4],
            "n_runs": len(grp), "cameras": ",".join(sorted(set(grp["cameras"].astype(str)))),
            "base_APm": grp["base_APm"].mean(), "final_APm": grp["APm_co"].mean(),
            "APG_m": grp["APG_m_co"].mean(), "APG_m_std": grp["APG_m_co"].std(ddof=0),
            "APG_50": grp["APG_50_co"].mean(), "APG_75": grp["APG_75_co"].mean(),
            "min_APG_m": grp["APG_m_co"].min(), "runtime_min": grp["runtime_min"].mean(),
        })
    by_method = pd.DataFrame(rows).sort_values("APG_m", ascending=False)
    (out_dir / "by_method.md").write_text("# Mean over cameras and seeds (APG = adapted − base)\n\n" + _md(by_method) + "\n")

    piv = df.pivot_table(index="exp", columns="cameras", values="APG_m_co", aggfunc="mean")
    piv["mean"] = piv.mean(axis=1)
    piv = piv.sort_values("mean", ascending=False).reset_index()
    (out_dir / "per_camera.md").write_text("# APG_m_co per camera (mean over seeds)\n\n" + _md(piv) + "\n")
    print(_md(by_method))
    print()
    print(_md(piv))
    print(f"\n{len(df)} rows -> {out_dir}")


if __name__ == "__main__":
    main()
