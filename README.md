# TCLD

**T**emporal **C**onsistency of **L**ocalization **D**istributions

Self-Supervised Adaptation Method for Detection Transformers Based on Temporal Consistency of Localization Distributions

Метод самоконтрольованої адаптації детекційного трансформера до умов експлуатації на основі темпоральної узгодженості розподілів локалізації

Master's thesis project. The research plan and progress log live in [`ai-plan.md`](ai-plan.md);
the thesis formulation is in `FORMULA.md` (read-only).

## What it does

Adapts a COCO-pretrained [D-FINE](https://github.com/Peterande/D-FINE) detector to a specific
static camera using only unlabeled video. D-FINE predicts a discrete distribution over bin
positions for each box edge; TCLD adds a self-supervised loss that enforces consistency of these
edge distributions for the same object across neighbouring frames (after motion compensation).
No labels, no architecture change, no inference cost. Evaluated on
[Scenes100](https://github.com/cvlab-stonybrook/scenes100) with the per-camera AP-gain protocol.

## Layout

```
external/D-FINE/   pinned upstream submodule (never edited in place)
tcld/              our package: model adapter, data, tcl loss, training, eval, analysis
configs/           hardware profiles (laptop / server), model configs, experiment configs
scripts/           download / run / aggregate entry points
tests/             CPU unit tests
results/           experiments.csv (source of truth), generated tables and figures
docker/            Dockerfile; docker-compose.yml at the root
```

## Quick start (Docker, recommended)

Requirements: Docker with the NVIDIA container toolkit.

```bash
git clone --recursive git@github.com:myrrhfawn/tcld.git && cd tcld
cp .env.example .env            # set TCLD_DATA / TCLD_RUNS / TCLD_CKPT (host paths)
make docker-build               # builds tcld:latest with your UID/GID
make docker-run CMD="make download-ckpt"           # D-FINE S + L checkpoints -> $TCLD_CKPT
make docker-run CMD="make download-coco-subset"    # COCO val annotations + 50 images -> $TCLD_DATA/coco
make docker-run CMD="make test"                    # CPU unit tests
make docker-run CMD="make smoke PROFILE=laptop MODEL=dfine_s"
```

Data, checkpoints and run outputs live in the mounted volumes (`/data`, `/ckpt`, `/runs` inside
the container) and are never baked into the image. On a new machine: clone, write `.env`, build,
mount the same volumes.

`make docker-shell` opens an interactive shell in the container.

## Quick start (host, no Docker)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
make setup
make download-ckpt download-coco-subset
make test
make smoke PROFILE=laptop MODEL=dfine_s
```

## Profiles

| profile  | target                     | model    | notes                                      |
|----------|----------------------------|----------|--------------------------------------------|
| `laptop` | RTX 3060 Laptop, 6 GB      | D-FINE-S | AMP, batch 2, frozen backbone + encoder    |
| `server` | multi-GPU server           | D-FINE-L | AMP, batch 8, full fine-tuning             |

## Reproducibility rules

* every experiment = a config under `configs/exp/` + a seed + a row in `results/experiments.csv`;
* tables under `results/tables/` are regenerated with `make aggregate`, never edited by hand;
* D-FINE is pinned by the submodule commit; see `ai-plan.md` for the full protocol.
