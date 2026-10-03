"""Rebuild results/tables/*.md from results/experiments.csv.

Skeleton (P0.6). Real aggregation (APG, Wilcoxon, main/ablation tables) lands in P6.4.
"""

from __future__ import annotations

import pandas as pd

from tcld.paths import RESULTS


def main() -> None:
    csv = RESULTS / "experiments.csv"
    df = pd.read_csv(csv)
    out_dir = RESULTS / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        print(f"{csv}: no experiments yet")
        return
    cols = ["exp_id", "phase", "method", "model", "profile", "cameras", "seed",
            "APm_co", "AP50_co", "AP75_co", "APG_m_co", "APG_50_co"]
    table = df[[c for c in cols if c in df.columns]].to_markdown(index=False)
    (out_dir / "all_experiments.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
