# TCLD

**T**emporal **C**onsistency of **L**ocalization **D**istributions: self-supervised per-camera adaptation of a
detection transformer from unlabeled video.

Master's thesis project. The thesis formulation is in `FORMULA.md`; the formal description of the method
(bin grids, transport between grids, motion compensation, properties of the W1 term, anchor, applicability
conditions, mapping to the code) is in [`docs/method_formal_2026-10-26.md`](docs/method_formal_2026-10-26.md).

## Method in one paragraph

[D-FINE](https://github.com/Peterande/D-FINE) predicts, for every box edge, a discrete distribution over 33 bin
positions instead of a single coordinate. On a static camera the same object seen in two nearby frames must
have the same edge distributions once its motion is compensated. TCLD adapts a COCO-pretrained D-FINE to one
camera by (1) matching detections of the frozen initial model across a frame pair (Hungarian assignment on IoU
after motion compensation, same class, confidence and IoU gates), (2) mapping both edge distributions to
absolute pixel coordinates (every query has its own bin grid), (3) minimising the 1-D Wasserstein-1 distance
between the two distributions on the union grid, symmetrically with stop-gradient on the target side, and
(4) anchoring class scores and boxes to the frozen initial model on the same frames so the detector cannot
drift. No labels, no architecture change, no inference cost. Final recipe:
`configs/exp/tf_a03_w5_n32_iou08_dense.yml` (TCLD weight 5, anchor 0.3, IoU gate 0.8, at least 32 pairs per
step, hybrid motion compensation = frozen-model shift with optical-flow fallback, 750 iterations, lr 1e-5,
EMA 0.9995, decoder trainable, backbone and encoder frozen). The control arm
`configs/exp/ctrl_a03_dense.yml` is the same file with `tcl_weight: 0.0`; every other arm differs from the
method in exactly one factor (checked by `scripts/audit_protocol.py`).

## Layout

```
external/D-FINE/      pinned upstream detector (git submodule, never edited)
external/scenes100/   pinned Scenes100 code (git submodule) - evaluation protocol reference only
tcld/model/           D-FINE adapter (distribution-mode forward), EMA, AdaBN
tcld/data/            MOT / VIRAT loaders, frame-pair sampler, synthetic shifts (night, jpeg)
tcld/tcl/             grid.py (bin geometry, transport), motion.py, matching.py, loss.py (W1 / KL / JS),
                      collapse.py (anchor, diagnostics), sharpness.py
tcld/train/           criterion.py (TCLD and the coordinate / moment / same-frame variants),
                      adapt.py (per-camera loop), self_train.py (self-training, Mean Teacher),
                      tta.py (Tent, ActMAD)
tcld/eval/            two-class (person / vehicle) COCO evaluation with ignore regions (Scenes100 protocol)
configs/              profiles (laptop / server), model configs, experiment configs (configs/exp)
scripts/              run_adapt.py, run_eval.py, sweep_dev.sh, reproduce.sh, stats_*.py, abs_table.py,
                      audit_protocol.py, make_paper_configs.py, download_*
tests/                CPU unit tests (geometry, transport, loss, matching, motion, data, eval, baselines)
```

The lab notebook and the supervisor reports are kept outside this repository.

## Setup (Docker)

Requirements: Docker with the NVIDIA container toolkit, a GPU with 6 GB or more.

```bash
git clone --recursive https://github.com/myrrhfawn/tcld.git && cd tcld
cp .env.example .env            # host paths for data / runs / checkpoints
make docker-build               # pytorch 2.8 + CUDA 12.8 image, your UID/GID
make docker-run CMD="make download-ckpt"           # dfine_s_coco.pth, dfine_l_coco.pth -> $TCLD_CKPT
make docker-run CMD="make download-coco-subset"    # COCO val2017 annotations + 50 images (smoke test)
make docker-run CMD="python scripts/download_coco_replay.py"   # 500 val2017 images (ActMAD source moments)
make docker-run CMD="make test"                    # CPU unit tests
make docker-run CMD="make smoke PROFILE=laptop MODEL=dfine_s"
```

Without Docker: install a CUDA build of torch 2.8, then `make setup` and the same `make` targets.

## Data

Place the datasets under `$TCLD_DATA` (host path from `.env`, mounted as `/data` in the container):

```
$TCLD_DATA/mot/MOT17Det/train/MOT17-02/{img1/, gt/gt.txt, seqinfo.ini}   MOT17Det.zip from motchallenge.net
$TCLD_DATA/mot/MOT20/train/MOT20-01/...                                    MOT20.zip from motchallenge.net
$TCLD_DATA/virat/videos/**/VIRAT_S_*.mp4                                   VIRAT Video Dataset Release 2.0
$TCLD_DATA/virat/ann/annotations/*.viratdata.objects.txt                   (data.kitware.com, collection VIRAT)
$TCLD_DATA/coco/val2017/*.jpg                                              see download scripts above
```

Static-camera MOT sequences used: MOT17-02, -04, -09 and MOT20-01, -02, -03, -05 (person class). VIRAT scenes
used: 0000 0001 0002 0100 0101 0102 0400 0401 0500 0502 0503 (person + vehicle). Only the official
annotations are used; MOT class 8 and 12 and VIRAT types 4 and 5 become ignore regions.

## Evaluation protocol

* Unit of analysis: the camera. Each cell adapts one model on the unlabeled frames of one camera and
  evaluates it on held-out frames of the same camera; metrics are COCO mAP@[.5:.95], AP50 and AP75 over the
  two target classes, Scenes100-style (ignore regions removed before matching).
* Original recordings: adapt on the first 70 % of frames, evaluate every 3rd frame of the remaining 30 %.
* Synthetic shifts: the same split with `corruption: night` (gamma darkening, sensor noise, JPEG q=30) or
  `corruption: jpeg` (1/3 scale, JPEG q=15, upscaled) applied to every decoded frame.
* Cross-recording: adapt on the unlabeled MOT *test* recording of a camera, evaluate on the whole labeled
  *train* recording (pairs eval:adapt MOT17-04:MOT17-03, MOT20-01:MOT20-07, MOT20-03:MOT20-04,
  MOT20-05:MOT20-04, MOT17-02:MOT17-01; `--adapt-seq`).
* VIRAT: adapt on 70 % of the clips of a scene, evaluate every 30th frame of the held-out clips.
* Seeds 0 1 2 for D-FINE-S, seed 0 for D-FINE-L; comparisons are paired per camera on common seeds
  (sign counts and exact Wilcoxon).

## Running

Everything below runs inside the container (`make docker-shell`, or prefix with `make docker-run CMD="..."`).

```bash
# zero-shot (unadapted) detector, one line per camera
python scripts/run_eval.py --dataset mot --profile laptop --model dfine_s --cameras MOT17-02 MOT17-04 --frame-stride 3

# one adaptation cell: method, camera MOT17-04, seed 0 (writes runs/<exp>/mot/result_MOT17-04_seed0.json)
python scripts/run_adapt.py --exp configs/exp/tf_a03_w5_n32_iou08_dense.yml --profile laptop \
    --dataset mot --cameras MOT17-04 --seed 0 --iters 750

# the paired control cell
python scripts/run_adapt.py --exp configs/exp/ctrl_a03_dense.yml --profile laptop \
    --dataset mot --cameras MOT17-04 --seed 0 --iters 750

# cross-recording cell
python scripts/run_adapt.py --exp configs/exp/tf_a03_w5_n32_iou08_dense_xrec.yml --profile laptop \
    --dataset mot --cameras MOT17-04 --adapt-seq MOT17-03 --seed 0 --iters 750

# D-FINE-L (batch 1, same number of pairs seen as S at 750 iterations)
python scripts/run_adapt.py --exp configs/exp/tf_a03_w5_n16_iou08_dense_L.yml --profile laptop \
    --dataset mot --cameras MOT17-04 --seed 0 --iters 1500

# a whole grid (skips cells whose result file exists): dataset, cameras, iters, experiments, seeds
bash scripts/sweep_dev.sh mot "MOT17-02 MOT17-04 MOT17-09 MOT20-01 MOT20-03 MOT20-05" 750 \
    "tf_a03_w5_n32_iou08_dense ctrl_a03_dense" "0 1 2"

# the complete protocol of the paper (method, control, baselines, encoder variants, cross-recording, L, VIRAT)
bash scripts/reproduce.sh all          # or a subset: bash scripts/reproduce.sh method baselines
```

Set `TCLD_NO_CKPT=1` (default in `reproduce.sh`) to skip saving adapted checkpoints. One S cell takes about
15 min on an RTX 3060 Laptop (6 GB), one L cell about 25 min.

## Tables

```bash
python scripts/audit_protocol.py                      # one-factor check of every arm, zero-shot consistency
python scripts/stats_baselines.py                     # per-camera deltas, sign counts, Wilcoxon: method vs baselines
python scripts/stats_method_vs_control.py --preset xrec5   # cross-recording; presets L and virat for the others
python scripts/abs_table.py                           # absolute mAP / AP50 / AP75 of every method per setting
```

Outputs go to `results/tables/`. Experiment configs for the VIRAT and D-FINE-L arms are derived from the
final method config by `scripts/make_paper_configs.py` rather than typed by hand.

## Experiment configs (`configs/exp`)

| family | config prefix | what differs from the method |
|---|---|---|
| TCLD (method) | `tf_a03_w5_n32_iou08_dense` | reference |
| control | `ctrl_a03_dense` | `tcl_weight: 0.0` |
| Tent | `tent_a03_dense` | entropy minimisation on normalisation affine parameters, no anchor |
| ActMAD | `actmad_a03_dense` | activation moment matching to COCO, no anchor |
| self-training | `st_a03_dense` | hard pseudo-boxes from the frozen model, upstream D-FINE criterion |
| Mean Teacher | `mt_a03_dense` | EMA teacher labels the clean view, student trained on a strong view |
| Mean Teacher + TCLD | `mttcl_w10_a03_dense` | Mean Teacher plus 10 x the TCLD term |
| coordinate consistency | `coord_a03_w5_n32_iou08_dense` | L1 + GIoU on decoded edges instead of W1 on distributions |
| moment consistency | `moments_a03_w5_n32_iou08_dense` | L1 on edge mean and std instead of W1 |
| same-frame two-view | `sameframe_a03_w5_n32_iou08_dense` | two photometric views of one frame, no temporal pair |
| D-FINE-L | `tf_a03_w5_n16_iou08_dense_L`, `ctrl_a03_dense_L` | model, batch 1, 16-pair floor |

Suffixes: `_night`, `_jpeg` (synthetic shift), `_xrec` (cross-recording), `_virat`, `_enc` (trainable encoder).
