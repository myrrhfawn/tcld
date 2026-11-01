"""Derive the paper-extension configs from the final method config (rule 15: one change per arm,
scaled keys read from the source file, never typed by hand).

  python scripts/make_paper_configs.py            # writes / verifies configs/exp/*_virat.yml and *_L.yml

VIRAT arm : same recipe, eval_frame_stride 30 (VIRAT protocol: every 30th frame of the eval clips).
D-FINE-L  : model dfine_l, batch 1 (6 GB), iters x2 (same number of pairs seen), N0 / 2 (same pairs
            per step floor).  Control arms = tcl_weight 0.0 only.
"""
import os
import sys

import yaml

ROOT = os.path.join(os.path.dirname(__file__), "..", "configs", "exp")
METHOD = "tf_a03_w5_n32_iou08_dense"


def load(name):
    with open(os.path.join(ROOT, name + ".yml")) as f:
        return yaml.safe_load(f)


def dump(name, cfg, header):
    path = os.path.join(ROOT, name + ".yml")
    if os.path.exists(path) and load(name) == cfg:
        print(f"unchanged  {name}")
        return
    with open(path, "w") as f:
        f.write("".join(f"# {h}\n" for h in header))
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"written    {name}")


def virat():
    base = load(METHOD)
    assert base["eval_frame_stride"] == 3, base["eval_frame_stride"]
    tf = dict(base, name=METHOD + "_virat", eval_frame_stride=30)
    dump(tf["name"], tf, ["VIRAT arm of the final method config: identical recipe, eval every 30th frame of the",
                          "eval clips (VIRAT protocol, FINDINGS 2026-10-03).", f"Derived from {METHOD} by scripts/make_paper_configs.py."])
    ctrl = yaml.safe_load(yaml.safe_dump(tf))
    ctrl["name"] = "ctrl_a03_dense_virat"
    assert ctrl["criterion"]["tcl_weight"] == 5.0
    ctrl["criterion"]["tcl_weight"] = 0.0
    dump(ctrl["name"], ctrl, ["Control for the VIRAT arm: tcl_weight 0.0 (anchor-only self-distillation), everything else",
                              f"identical to {tf['name']}.", "Derived by scripts/make_paper_configs.py."])


def dfine_l():
    for suf in ["", "_night", "_jpeg", "_xrec"]:
        src = load(METHOD + suf)
        n0, bs = src["criterion"]["tcl"]["min_denom_pairs"], src["batch_size"]
        assert (n0, bs) == (32, 2), (n0, bs)
        tf = yaml.safe_load(yaml.safe_dump(src))
        tf["name"] = f"tf_a03_w5_n16_iou08_dense{suf}_L"
        tf["model"] = "dfine_l"
        tf["batch_size"] = bs // 2
        tf["criterion"]["tcl"]["min_denom_pairs"] = n0 // 2
        # `iters` in the yml is a dead key: every queue passes --iters on the command line (S: 750, L: 1500,
        # i.e. the same number of pairs seen at half the batch).
        hdr = [f"D-FINE-L arm of {METHOD}{suf}: model dfine_l, batch {bs}->{bs // 2} (6 GB), N0 {n0}->{n0 // 2}",
               "(same pair floor per step); run with --iters 1500 (S: 750) so the number of pairs seen is equal.",
               "Everything else identical. Derived by scripts/make_paper_configs.py."]
        if suf == "_night" and os.path.exists(os.path.join(ROOT, tf["name"] + ".yml")):
            old = load(tf["name"])
            if old != tf:
                print(f"MISMATCH   {tf['name']} (existing night_L differs: "
                      f"{ {k: (old.get(k), tf.get(k)) for k in set(old) | set(tf) if old.get(k) != tf.get(k)} })")
                sys.exit(1)
            print(f"verified   {tf['name']}")
        else:
            dump(tf["name"], tf, hdr)
        ctrl = yaml.safe_load(yaml.safe_dump(tf))
        ctrl["name"] = f"ctrl_a03_dense{suf}_L"
        ctrl["criterion"]["tcl_weight"] = 0.0
        dump(ctrl["name"], ctrl, [f"Control for {tf['name']}: tcl_weight 0.0, everything else identical.",
                                  "Derived by scripts/make_paper_configs.py."])


if __name__ == "__main__":
    virat()
    dfine_l()
