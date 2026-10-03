# ai-plan.md — план виконання магістерської TCLD

> **TCLD** — Temporal Consistency of Localization Distributions.
> Метод самоконтрольованої адаптації детекційного трансформера (D-FINE) до умов експлуатації на основі темпоральної узгодженості дискретних розподілів локалізації між сусідніми кадрами.
> Джерело істини щодо теми, мети, завдань і новизни — `FORMULA.md`. Цей файл — робочий план і журнал; його оновлюють AI-агенти та автор.

---

## 0. Правила для AI-агентів (обов'язково прочитати перед будь-якою роботою)

1. **`FORMULA.md` — read-only.** Ніколи не редагувати. Якщо план або код розходяться з FORMULA.md — правити план і повідомити автора.
2. **Жодних `git commit` / `git push` / `git init` / rebase без явного дозволу автора в поточній сесії.** Дозвіл з минулої сесії не діє. Зміни у файлах робити можна, коміт — ні.
3. **Дані, чекпоінти, логи прогонів — поза git.** `data/`, `checkpoints/`, `runs/`, `external/*/outputs` у `.gitignore`. У git — код, конфіги, `results/experiments.csv`, `results/tables/*.md`, невеликі фігури (`results/figures/*.png|svg`, < 1 МБ кожна).
4. **Два профілі запуску, нуль remote-коду.** `configs/profiles/laptop.yml` (RTX 3060 Laptop 6 ГБ: D-FINE-S, AMP, batch 2, frozen backbone, 640 px) і `configs/profiles/server.yml` (mlgpu-сервер: D-FINE-L/M, batch ≥ 8). Профіль — аргумент CLI; шляхи — через змінні `TCLD_DATA`, `TCLD_RUNS`, `TCLD_CKPT`. Ніякого SSH, rsync, оркестрування — автор сам переносить на сервер і запускає там. Усе, що потрібне для прогону, має бути в репо.
4a. **Усе працює в Docker.** Середовище живе в образі `tcld` (`docker/Dockerfile`, `docker-compose.yml`): код, D-FINE, залежності, pinned версії. На новому сервері достатньо `docker compose build` (або `docker pull` з registry, якщо автор його підніме) + змонтувати `TCLD_DATA`/`TCLD_RUNS`/`TCLD_CKPT` як volumes. Дані, чекпоінти й результати — **ніколи не в образі**, тільки у volumes. Кожна команда з `Makefile` має працювати і всередині контейнера, і через `docker compose run tcld make <target>`. Будь-яка нова залежність додається у `requirements.txt` **і** перевіряється збіркою образу. Образ тегується `tcld:<git-short-sha>` плюс `tcld:latest`.
5. **Статуси задач:** `[ ]` не почато · `[~]` в роботі · `[x]` зроблено (критерій готовності виконано) · `[!]` заблоковано (причина в журналі). Задача має ID (`P4.3`). Перед тим як починати задачу, перевір її залежності.
6. **Кожна сесія роботи завершується записом у розділ 7 «Журнал виконання»**: дата, агент/модель, ID задач, що зроблено, де артефакти (шляхи), що не вийшло, наступний крок. Без запису в журналі робота вважається не зробленою.
7. **Кожен експеримент = конфіг + сід + рядок у `results/experiments.csv`.** Числа без конфігу і сіду не є результатом. Нейминг у розділі 5.
8. **Протокол оцінки (класи, маски, метрики) не змінювати** без запису в журнал і погодження з автором. Dev-сабсет камер фіксується до першого тренування і не перевибирається після результатів.
9. **Не ламати upstream.** D-FINE підключається як git submodule з pinned commit; зміни в його коді — тільки через monkey-patch/обгортки в пакеті `tcld/` або мінімальний патч-файл `external/patches/*.patch`, задокументований у журналі.
10. **Кожна нова функція ядра (`tcld/tcl/*`) має юніт-тест на синтетичних даних.** `make test` має проходити на CPU.
11. Коли щось не працює — пиши в журнал чесно: що пробував, яка помилка, що відклав. Невдалі спроби потрібні для тексту роботи.

---

## 1. Контекст і ключові факти

### 1.1. Мапінг завдань керівника (FORMULA.md) на фази плану

| Завдання керівника | Фази |
|---|---|
| 1. Аналіз методів донавчання детекторів за нерозміченим відео | P2 (бейзлайни як референс), P5 (реалізація бейзлайнів), результати P6 — фактичний матеріал для огляду |
| 2. Дослідити зміни розподілів локалізації одного об'єкта в послідовних кадрах | **P3** (кореляційне дослідження) |
| 3. Удосконалити метод донавчання на основі узгодження цих розподілів | **P4** (метод TCLD) |
| 4. Експериментально оцінити і порівняти з базовим детектором і відомими способами донавчання | P2 (zero-shot), P5 (бейзлайни), **P6** (абляції + full100), P7 (додаткові дані/детектори) |

### 1.2. D-FINE (ICLR 2025, Apache-2.0, `github.com/Peterande/D-FINE`)

- Декодер видає `pred_corners` форми `[B, Q, 4, reg_max+1]`, `reg_max=32`: 4 незалежні softmax-розподіли по бінах для 4 відстаней до країв бокса. Інтеграл `Integral` з нерівномірною ваговою функцією `weighting_function(reg_max, up, reg_scale)` (`up=0.5`, `reg_scale=4.0`) відносно reference point дає відстані; `distance2bbox` → бокс.
- **Критично:** усі шари декодера уточнюють corner-логіти відносно одного й того самого `ref_points_initial` (encoder top-k пропозал). Бін `n` лежить в абсолютних координатах на `edge_ref + (0.5·reg_scale + W(n)) · ref_size / reg_scale`. W(n) нерівномірна — щільна біля нуля, груба в хвостах. Саме тому GO-LSD може порівнювати шари напряму. **Між кадрами top-k пропозали різні → сітки бінів відрізняються і зсувом, і шириною. Сирий KL між `pred_corners` двох кадрів безглуздий.**
- Лосси в `src/zoo/dfine/dfine_criterion.py`: `loss_fgl` (unimodal distribution focal loss vs GT, через `bbox2distance`), `loss_ddf` (KL з T=5 від `teacher_corners` фінального шару до мілкіших — GO-LSD), VFL, L1 + GIoU. `loss_local` рахує FGL/DDF **лише на Hungarian-індексах** → для крос-кадрових пар потрібен власний локальний лосс.
- Output у train-режимі: `pred_logits`, `pred_boxes`, `pred_corners`, `ref_points`, `up`, `reg_scale`, `aux_outputs`, `teacher_corners`, `teacher_logits`. В eval — тільки `pred_logits`, `pred_boxes`. ⇒ Для отримання розподілів обидва кадри і teacher ганяти в `model.train()` із замороженими norm-шарами.
- Denoising-групи будуються з таргетів (`get_contrastive_denoising_training_group`) → для нерозмічених батчів потрібен гард `targets=None` → DN пропускається (модель збіжна, це безпечно).
- Чекпоінти: `https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_{n,s,m,l,x}_coco.pth`, `dfine_{s,m,l,x}_obj2coco.pth`, `dfine_{s,m,l,x}_obj365.pth`. COCO AP: N 42.8 / S 48.5 / M 52.3 / L 54.0 / X 55.8.
- Тренування: `train.py -c configs/dfine/dfine_hgnetv2_{s,l}_coco.yml --use-amp --seed=0`, кастомний датасет через `configs/dataset/custom_detection.yml` (`num_classes`, `remap_mscoco_category: False`).
- Follow-up: DEIM (CVPR 2025, `github.com/ShihuaHuang95/DEIM`) — тренувальний фреймворк поверх D-FINE (Dense O2O + MAL), кандидат для P7.

### 1.3. Scenes100 (CVPR 2023, MIT, `github.com/cvlab-stonybrook/scenes100`)

- 100 відео зі статичних YouTube-камер (live streams), ≥ 2 год кожне, ≥ 720p, 16 країн, день/ніч, indoor/outdoor. Перші 1.5 год — нерозмічений train; решта — eval (рівномірно семпловані вручну анотовані кадри, кількість обернено пропорційна щільності об'єктів; ~840 боксів/відео).
- **2 класи:** `person` (COCO person) і `vehicle` (COCO car + bus + truck). Усе інше відкидається. Non-evaluation полігональні маски виключають далекі зони: бокс, у якого хоч один кут у масці, вилучається і з GT, і з детекцій перед COCO-evaluator.
- Метрики: `APm_co`, `AP50_co` (COCO mean по 2 класах) і `APm_w`, `AP50_w` (зважені за частотою класів у val). Головна метрика — **APG = (1/100) Σ_v (AP_v,adapt − AP_v,base)**, окрема модель на кожне відео.
- Референсні числа (Faster R-CNN R-101, M2): база на Scenes100 APm_co 41.11 / AP50_co 63.10 (на COCO val 51.29 / 77.46).
  APG^m_co / APG^50_co: **ST +0.80 / +0.24**, STAC −1.26 / −5.12, AT −0.75 / −1.11, H2FA −3.10 / −4.97, TIA −0.32 / −0.37, **LODS +0.45 / +1.28**, **тільки псевдомітки (без mixup/fusion) +0.95 / +0.54**, **повний метод +3.76 / +4.45**. LODS деградує при > 1000 ітерацій. Абсолютні AP D-FINE з ними непорівнянні — порівнюємо лише прирости і перезапускаємо їхні псевдомітки на D-FINE.
- Їхні псевдомітки: детекції двох базових моделей (conf > λ_det) + DiMP-50 трекінг вперед/назад (≤ 2 с, PyTracking commit 47d9c16), графова дедуплікація за IoU. У train-батчі домішують стільки ж COCO-зображень, скільки нерозмічених кадрів.
- Код: `datasets.py --opt download --target {annotation,video,image} [--ids 003]`, `pseudo_label.py`, `evaluate_adaptation.py --opt single --id 001 ...`. Повний датасет ~2.2 ТБ (pre-extracted frames ×3 від відео). 20 ГБ VRAM потрібно тільки їхньому fusion-методу — нам ні.

### 1.4. Додаткові дані

- **UA-DETRAC** (статичні дорожні камери, щільний GT) — офіційний сайт завантаження фактично недоступний; перевірити дзеркала. Fallback для щільного GT: **MOT17** статичні послідовності (02, 04, 09) та **MOT20**.
- **COCO train2017/val2017** — для replay-сабсету і sanity-метрики (COCO-minival-500).

### 1.5. Залізо

| Профіль | GPU | Модель | Батч | Призначення |
|---|---|---|---|---|
| `laptop` | RTX 3060 Laptop 6 ГБ | D-FINE-S (fallback N) | 2 пари, AMP, frozen backbone під `no_grad`, 640 px (fallback 512) | розробка, юніт-тести, dev5-експерименти |
| `server` | mlgpu (багато VRAM) | D-FINE-L (M для абляцій) | ≥ 8 | full100, dev10×3 сіди, Scenes100-PL генерація |

---

## 2. Цільова структура репозиторію

```
tcld/
  FORMULA.md                 # read-only
  README.md  ai-plan.md
  pyproject.toml  requirements.txt  Makefile  .gitignore  .dockerignore  .env.example  .gitmodules
  docker/Dockerfile  docker-compose.yml   # усе середовище; дані/чекпоінти/результати — volumes
  external/
    D-FINE/                  # git submodule, pinned commit
    scenes100/               # git submodule (тільки для datasets.py / pseudo_label.py / evaluate_adaptation.py)
    patches/                 # мінімальні .patch для upstream, якщо без них ніяк
  tcld/                      # власний пакет
    __init__.py
    data/
      scenes100.py           # download wrapper, конвертація eval-анотацій у COCO-формат (2 класи), non-eval маски, dev-сабсети
      pairs.py               # frame-pair sampler (t, t+Δ) з детермінованим сідом
      clips.py               # декодування коротких кліпів навколо timestamp'ів
      coco_replay.py         # COCO remapped сабсет + minival-500
      mot.py                 # MOT17/MOT20/UA-DETRAC loader для P3
    model/
      dfine_adapter.py       # обгортка D-FINE: build з чекпоінту, forward → corners/ref_points/boxes/logits, DN-guard, freeze
      base.py                # абстрактний інтерфейс DistributionalDetector (для P7: GFL, DEIM, ...)
      ema.py
    tcl/
      grid.py                # абсолютні позиції бінів, CDF-ресемплінг розподілу на іншу сітку, W1
      motion.py              # компенсація руху: predicted-center / optical flow / frozen-model
      matching.py            # крос-кадровий матчинг детекцій
      sharpness.py           # диференціальна ентропія в пікселях, дисперсія
      loss.py                # TCL-лосс (KL/W1, асиметрія, ваги)
      collapse.py            # регуляризатори (якір, entropy floor) + діагностика
    train/
      criterion.py           # TCLDCriterion: supervised частина (COCO replay) + TCL + якір
      adapt.py               # per-camera цикл адаптації
      baselines/
        self_train.py  coord_consistency.py  moments.py  same_frame_kl.py  scenes100_pl.py
    eval/
      scenes100_eval.py      # APm/AP50 co/w з non-eval масками (еквівалент evaluate_adaptation.py)
      apg.py                 # агрегація APG, Wilcoxon, таблиці
      coco_sanity.py
      speed.py               # FPS / ONNX-експорт — підтвердження, що інференс не змінився
    analysis/
      distributions.py       # KL/W1/ентропія vs похибка (P3)
      plots.py
  configs/
    profiles/laptop.yml  profiles/server.yml
    model/dfine_s.yml  dfine_l.yml
    data/scenes100_dev5.yml  scenes100_dev10.yml  scenes100_full100.yml  mot_static.yml
    exp/                     # один yml на експеримент
  scripts/
    download_dfine_ckpt.sh  download_scenes100.py  download_mot.sh  download_coco_subset.py
    run_adapt.py  run_eval.py  run_study.py  aggregate.py
  notebooks/01_distribution_study.ipynb  02_results.ipynb
  results/experiments.csv  tables/  figures/
  tests/                     # pytest, CPU-only, синтетика
```

---

## 3. Фази та задачі

### P0 — Інфраструктура

- [ ] **P0.1** `requirements.txt` / `pyproject.toml` (torch+CUDA версія під обидві машини, torchvision, pycocotools, opencv, scipy, pandas, matplotlib, pytest; версії зафіксовані), `.gitignore`, `.dockerignore`, `.env.example` з `TCLD_DATA/TCLD_RUNS/TCLD_CKPT`.
- [ ] **P0.1a** **Docker.** `docker/Dockerfile` на базі офіційного `pytorch/pytorch:<ver>-cuda<ver>-cudnn-runtime` (або `nvidia/cuda` + pip), з `ffmpeg`, `libgl1`, `git`; non-root user з UID/GID хоста (щоб файли у volumes не ставали root-овими); `pip install -r requirements.txt`; `pip install -e .`; D-FINE submodule всередині образу; `PYTHONPATH` налаштований. `docker-compose.yml`: сервіс `tcld` з `gpus: all`, `shm_size: 8g`, volumes `${TCLD_DATA}:/data`, `${TCLD_RUNS}:/runs`, `${TCLD_CKPT}:/ckpt`, `.:/workspace` (dev-режим — код монтується, не копіюється), env з `.env`. `Makefile` targets: `docker-build`, `docker-shell`, `docker-run CMD=...`. Перевірка: образ збирається на ноуті, `docker compose run tcld make smoke` проходить, `nvidia-smi` видно з контейнера. Записати в `README.md` розділ «Запуск через Docker» (новий сервер: clone → `.env` → `docker compose build` → `make docker-run CMD="make adapt ..."`).
- [ ] **P0.2** D-FINE як submodule в `external/D-FINE` на pinned commit; `scripts/download_dfine_ckpt.sh` (S, L COCO; опційно obj2coco) — качає у `/ckpt` (volume), не в образ. Записати commit hash у журнал.
- [ ] **P0.3** `tcld/model/dfine_adapter.py`: завантаження моделі з yml+pth, `forward(images, mode="train"|"eval")`, що повертає `pred_corners`, `ref_points_initial`, `up`, `reg_scale`, `pred_boxes`, `pred_logits` для всіх шарів; freeze backbone/encoder; norm-шари у eval-режимі при `model.train()`; DN-guard при `targets=None`.
- [ ] **P0.4** Smoke-test: інференс на 50 зображеннях COCO val, AP збігається з README D-FINE у межах шуму на 50 зображеннях (порівняти з офіційним `train.py --test-only` на тих самих 50). `make smoke`.
- [ ] **P0.5** Профілі `laptop.yml` / `server.yml`, `Makefile` (`setup`, `smoke`, `test`, `study`, `adapt`, `eval`, `aggregate`), `tests/` каркас.
- [ ] **P0.6** `results/experiments.csv` з заголовком (схема в розділі 5), `scripts/aggregate.py` скелет.

**Критерій готовності P0:** `make smoke` і `make test` проходять на ноуті **всередині Docker-контейнера**; образ збирається з чистого clone; журнал містить версії torch/CUDA, базовий образ і commit D-FINE.

### P1 — Дані

- [ ] **P1.1** `scripts/download_scenes100.py`: обгортка над `external/scenes100/datasets.py`; спочатку завантажити **тільки анотації** для всіх 100 відео і перевірити, які відео/pre-extracted frames ще доступні у сховищі авторів. Записати список доступних у `configs/data/scenes100_available.yml` і в журнал.
- [ ] **P1.2** `tcld/data/scenes100.py`: конвертація eval-анотацій у COCO-json з 2 класами (`person`=COCO 1; `vehicle`=COCO 3,6,8), застосування non-evaluation масок **точно як у `evaluate_adaptation.py`** (бокс вилучається, якщо хоч один кут у масці — і для GT, і для детекцій). Юніт-тест на маски і mapping.
- [ ] **P1.3** `tcld/eval/scenes100_eval.py`: per-video `APm_co, AP50_co, APm_w, AP50_w`; валідація: на одному відео числа збігаються з `evaluate_adaptation.py` для одного й того ж файлу детекцій (±0.05).
- [ ] **P1.4** Стратегія зберігання: **не тримати 2.2 ТБ.** `tcld/data/clips.py`: з train-частини (перші 1.5 год) декодувати короткі кліпи на нативному fps навколо семплованих timestamp'ів (для пар з Δ∈{1,2,5,10,30} кадрів) + розріджені кадри 1 fps для ST/PL-бейзлайнів. Ціль ≤ 5–10 ГБ/камера. Параметри семплінгу — в `configs/data/*.yml`.
- [ ] **P1.5** `tcld/data/pairs.py`: детермінований sampler пар `(t, t+Δ)`, повертає обидва кадри + метадані (camera, timestamp, Δ); однакові аугментації на пару (тільки фотометричні; геометричні — ні, або однакові для обох).
- [ ] **P1.6** `tcld/data/coco_replay.py`: COCO train2017 remapped на 2 класи, сабсет 500–5k зображень (сід фіксований); COCO-minival-500 для sanity.
- [ ] **P1.7** `tcld/data/mot.py` + `scripts/download_mot.sh`: MOT17 (02, 04, 09 — статичні) і MOT20 у єдиному форматі (frame, track_id, box, class→person); перевірка доступності UA-DETRAC (дзеркала) — результат у журнал; якщо доступний — loader (vehicle).
- [ ] **P1.8** Dev-сабсет **після P2.1**: `configs/data/scenes100_dev5.yml` — 5 камер, стратифіковані за терцилем zero-shot AP і типом сцени (день/ніч, indoor/outdoor, щільність), **тільки з доступних**; `scenes100_dev10.yml` — 10 камер для серверної дисперсії. Список фіксується один раз; зміна — лише через журнал і погодження.

**Критерій готовності P1:** eval базової моделі на будь-якому доступному відео дає per-video AP; юніт-тести масок/mapping/sampler проходять; dev5/dev10 зафіксовані.

### P2 — Zero-shot бейзлайн і reference-метрики

- [ ] **P2.1** Zero-shot D-FINE-S (ноут) на всіх доступних камерах: `APm_co, AP50_co, APm_w, AP50_w, AP75` per-video → `results/tables/zero_shot_s.md`, рядки в csv. Це вхід для вибору dev5 (P1.8).
- [ ] **P2.2** Zero-shot D-FINE-L (сервер) на full100 → `zero_shot_l.md`.
- [ ] **P2.3** `tcld/eval/speed.py`: FPS на ноуті (PyTorch fp16) + ONNX-експорт штатним `tools/deployment/export_onnx.py` з оригінального і (пізніше) адаптованого чекпоінту — підтвердження, що граф ідентичний. Reference-числа в журнал.
- [ ] **P2.4** COCO-minival-500 AP базових моделей (sanity reference).

**Критерій готовності P2:** таблиці zero-shot для S і L, reference FPS, записи в csv.

### P3 — Дослідження розподілів між кадрами (завдання 2 керівника; gate для P4)

- [ ] **P3.1** `tcld/tcl/grid.py` (перша версія без градієнтів): абсолютні позиції меж бінів кожного краю (`ref_points_initial`, `ref_size`, `reg_scale`, `W(n)`), CDF, ресемплінг розподілу на довільну сітку, W1 між двома розподілами в пікселях.
- [ ] **P3.2** `tcld/tcl/sharpness.py`: **диференціальна ентропія в пікселях** `H_diff = H_bins + Σ p_n · log(width_n)` і дисперсія в пікселях. (Shannon-ентропія по бінах непорівнянна між сітками різного масштабу — не використовувати як міру гостроти.)
- [ ] **P3.3** `tcld/analysis/distributions.py` + `scripts/run_study.py`: для датасетів з щільним GT (MOT17-static, MOT20, UA-DETRAC якщо є) і Δ∈{1,2,5,10,30}: детекції базової моделі → матчинг до GT-треків → для пар одного об'єкта: KL і W1 між розподілами країв в абсолютних координатах після компенсації руху (три варіанти: GT-зсув, predicted-center, optical flow), H_diff кожного, похибки локалізації |edge−gt_edge| та IoU.
- [ ] **P3.4** Аналіз (`notebooks/01_distribution_study.ipynb`, фігури `results/figures/study_*`):
  - Spearman «розбіжність (KL, W1) ↔ max похибка пари» по Δ;
  - Spearman «H_diff ↔ похибка»;
  - **precision правила «гостре вчить розмите»**: частка пар, де гостріший розподіл є точнішим (по краях і по IoU), по Δ і величині руху;
  - розподіл KL/W1 для «правильних» vs «неправильних» пар; залежність від класу, розміру об'єкта, величини руху;
  - порівняння компенсації руху GT vs predicted-center vs flow (похибка після компенсації).
- [ ] **P3.5** Sanity на Scenes100 val-кадрах (sparse GT) — те саме, де можливо.
- [ ] **P3.6** Висновок у журнал і `results/tables/study_summary.md`: чи сигнал інформативний, яка компенсація руху найкраща, який Δ-діапазон робочий, яка метрика (KL vs W1) краще корелює з похибкою. **Це gate: P4.4 налаштовується за цими висновками.**

**Критерій готовності P3:** таблиця кореляцій + precision правила + явний висновок; фігури готові до вставки в роботу.

### P4 — Метод TCLD

- [ ] **P4.1** `tcld/tcl/grid.py` (диференційована версія): CDF на межах бінів у абсолютних пікселях → застосувати рух (трансляція + опційно масштаб відносно центру) → лінійна інтерполяція на межі бінів цільового кадру → різниця CDF = маси. **fp32 навіть під AMP.** Маса поза носієм цільової сітки: перенормувати, якщо втрачено < 10–20 %, інакше пара відкидається; частка відкинутих логується. Також W1 (L1 між CDF на об'єднаній сітці). Юніт-тести: тотожність при нульовому русі на тій самій сітці; збереження середнього при трансляції; `gradcheck`; інваріантність до зсуву сітки.
- [ ] **P4.2** `tcld/tcl/motion.py` — **ключове рішення проти колапсу на статичний фон.** На статичній камері край, що «сів» на статичну структуру фону, ідеально узгоджений між кадрами; якщо зсув брати з predicted-центрів, дрейфуючий бокс має нульовий зсув і нульовий лосс. Тому зсув береться з джерела, яке лосс не може «зіграти»: (а) оптичний потік у боксі (RAFT-small з torchvision або Farneback; медіана потоку в боксі), (б) бокси замороженої початкової моделі. Плюс: гейт за мінімальним зсувом (пари майже без руху не навчають) і вага за величиною зсуву. Варіант predicted-center — лише абляція.
- [ ] **P4.3** `tcld/tcl/matching.py`: Hungarian по IoU після компенсації + схожість класових логітів (cosine) серед детекцій з conf > τ; пороги τ, ρ(IoU) у конфігу; для Δ > 1 — матчинг через треки замороженої моделі (простий IoU-трекер). Повертає пари індексів запитів `(q_t, q_{t+Δ})` + ваги.
- [ ] **P4.4** `tcld/tcl/loss.py`: TCL = KL(p_sharp ‖ p_blurry) **або** W1 з stop-grad на гострішій стороні (гострота за P3.2), температура T, ваги = f(conf, |зсув|); режими: `asym` (sharp→blurry), `sym` (JS), `ema` (ціль від EMA-teacher на кадрі t, гострота — вага пари). Ціль — **тільки фінальний шар декодера** за замовчуванням (усі шари — абляція, перекривається з DDF). **Не застосовувати до encoder top-k голів** (там лише бокси).
- [ ] **P4.5** `tcld/tcl/collapse.py`:
  - обов'язкові: stop-grad асиметрія; conf+IoU гейт; **якір до замороженої початкової моделі на тих самих нерозмічених кадрах** (KL на класових логітах + GIoU на боксах; дешевше за COCO replay);
  - рекомендовано: EMA-teacher (`tcld/model/ema.py`), що дає розподіли кадру t;
  - опційно: entropy floor, COCO replay (P1.6);
  - **діагностика** кожні N кроків на фіксованому probe-сеті (50 кадрів/камера): середня H_diff по краях; частка країв з max p > 0.9; середнє відношення w/h боксів і |Δbox| до замороженої моделі; кількість детекцій > 0.3 і гістограма max-score; кількість matched пар та їх IoU; частка маси поза носієм; значення кожного лоссу; AP на COCO-minival-500 і на val-кадрах камери. Логи в `runs/<exp>/diag.csv` + tensorboard.
  - режими колапсу, які треба ловити: колапс на статичний фон; over-sharpening до спільних хибних країв (систематичне стиснення/розширення боксів); дрейф класифікації; uniform-колапс (малоймовірний при асиметрії).
- [ ] **P4.6** `tcld/train/criterion.py`: власний локальний лосс на крос-кадрових парах індексів (не через Hungarian); DDF лишається (без GT); supervised частина для COCO-replay кроків — штатний `DFINECriterion`. `tcld/train/adapt.py`: per-camera цикл від COCO-чекпоінту; чергування unlabeled/labeled кроків (не змішувати в батчі); `targets=None` → DN пропускається; `model.train()` з frozen norm; frozen backbone (+encoder) під `no_grad` у `laptop`; AMP; gradient checkpointing декодера за потреби; збереження чекпоінту + eval у кінці; `scripts/run_adapt.py --profile laptop --exp configs/exp/x.yml --camera 003 --seed 0`.
- [ ] **P4.7** Перший повний прогін на 1 dev-камері на ноуті: без NaN, діагностика пишеться, частка відкинутих пар залогована, AP ≥ zero-shot (або чесний запис чому ні).

**Критерій готовності P4:** P4.7 виконано, `make test` зелений, результат у csv.

### P5 — Бейзлайни на D-FINE (спільні матчинг/гейт/компенсація; різні лосси)

- [ ] **P5.1** `self_train.py` — ST: жорсткі псевдобокси замороженої/EMA-моделі з conf > τ як таргети у штатний criterion.
- [ ] **P5.2** `coord_consistency.py` — jitter-аналог: L1 + GIoU між `pred_boxes` сусідніх кадрів після компенсації, без розподілів (ізолює «розподіл vs точка»).
- [ ] **P5.3** `moments.py` — консистентність тільки середнього/дисперсії краю в пікселях (uncertainty-weighted jitter).
- [ ] **P5.4** `same_frame_kl.py` — два аугментованих view одного кадру, KL на розподілах (аналог Humble Teacher / LD без крос-кадру) — відповідь на «а чи важливий саме крос-кадр».
- [ ] **P5.5** `scenes100_pl.py` — псевдомітки методу Scenes100 (згенерувати їхнім `pseudo_label.py` на сервері для dev5/dev10/full100, або взяти готові, якщо викладені) → штатний criterion на D-FINE.
- [ ] **P5.6** TCLD + Scenes100-PL комбіновано (очікування: найкраще; регресійний self-supervision рухає переважно AP75, PL — recall; так і репортити).

**Критерій готовності P5:** кожен бейзлайн — 1 прогін на dev5 × 1 сід, рядки в csv.

### P6 — Експерименти та абляції

- [ ] **P6.1** Dev5 × 3 сіди (ноут, D-FINE-S): zero-shot, ST, coord-consistency, moments, same-frame KL, Scenes100-PL, TCLD, TCLD+PL. Парний дизайн: per-camera дельти, Wilcoxon signed-rank у `apg.py`.
- [ ] **P6.2** Абляції TCLD на dev5 × 3 сіди: Δ ∈ {1,2,5,10}; компенсація руху off / predicted-center / flow / frozen-model; KL vs W1; `asym` vs `sym` vs `ema`; мінус кожен регуляризатор (−anchor, −asymmetry, −motion-gate, −conf-gate); T; вага TCL; frozen vs full fine-tune; фінальний шар vs усі; крива AP vs iters (деградація після N ітерацій, як у LODS).
- [ ] **P6.3** Сервер, D-FINE-L: dev10 × 3 сіди для оцінки дисперсії; full100 × 1 сід для zero-shot, ST, coord-consistency, same-frame KL, Scenes100-PL, TCLD, TCLD+PL.
- [ ] **P6.4** `results/tables/main.md` у форматі Table 3 Scenes100 (`APG^m_co, APG^50_co, APG^m_w, APG^50_w`) **плюс окремо AP50 і AP75**; `ablations.md`; фігури (per-camera дельти, крива AP vs iters, діагностика колапсу).
- [ ] **P6.5** `speed.py` на адаптованому чекпоінті: FPS і ONNX-граф ідентичні базовим (підтвердження «інференс без змін»).

**Критерій готовності P6:** усі таблиці генеруються `scripts/aggregate.py` з `experiments.csv`; кожен рядок має конфіг і сід.

### P7 — Опціонально (після P6)

- [ ] **P7.1** UA-DETRAC покамерно (якщо доступний) або MOT20 як другий бенчмарк — той самий протокол (train-частина без міток, eval на GT).
- [ ] **P7.2** Інші детектори через `tcld/model/base.py`: DEIM-D-FINE (той самий FDR, інше тренування), GFL/GFLv2 (DFL-розподіли — природний другий носій), RT-DETRv2 (без розподілів — зафіксувати як обмеження або додати DFL-голову).
- [ ] **P7.3** «Propagated queries»: бокси кадру t після компенсації як `ref_points_unact` частини запитів кадру t+Δ (MOTR-подібно) — однакові сітки і матчинг безкоштовно; змінює forward → лише як розширення.
- [ ] **P7.4** Compound-режим: одна модель на всі камери (порівняння з per-camera).

### P8 — Фіналізація

- [ ] **P8.1** `scripts/aggregate.py` генерує всі таблиці/фігури з csv одним запуском; `notebooks/02_results.ipynb`.
- [ ] **P8.2** `README.md`: відтворення від нуля через Docker (clone → `.env` → `docker compose build` → download → study → adapt → eval → aggregate), версії, сіди, список доступних камер, обмеження; фінальний образ `tcld:<sha>` зафіксований у журналі.
- [ ] **P8.3** Архів фінальних конфігів `configs/exp/final/*`, чекпоінти найкращих моделей (поза git, шляхи в журналі).
- [ ] **P8.4** `results/limitations.md`: перелік невдалих спроб, обмежень, нерозв'язаних питань — матеріал для тексту роботи.

---

## 4. Порядок виконання і залежності

```
P0 → P1.1–P1.7 → P2.1 → P1.8 → P2.2–P2.4
P1.7 + P3.1–P3.2 → P3.3–P3.6  (gate)
P3.6 → P4.1–P4.7 → P5 → P6.1–P6.2 (ноут) → P6.3 (сервер) → P6.4–P6.5 → P7 → P8
```
P3 і P4.1–P4.3 можна вести паралельно; P4.4 остаточно налаштовується після P3.6.

---

## 5. Протокол експериментів

**Схема `results/experiments.csv`:**
`exp_id, phase, method, model, profile, cameras, seed, iters, delta, motion, loss_variant, reg, APm_co, AP50_co, AP75_co, APm_w, AP50_w, APG_m_co, APG_50_co, APG_75_co, coco_minival_AP, runtime_min, gpu, dfine_commit, config_path, notes`

**Нейминг:** `P6_tcld_s_dev5_d1_flow_asym_s0` = фаза, метод, модель, сабсет, Δ, компенсація руху, варіант лоссу, сід. Один yml на експеримент у `configs/exp/`.

**Сіди:** dev5 — 3 сіди (0, 1, 2) обов'язково; full100 — 1 сід (як у Scenes100); dev10 на сервері — 3 сіди.

**Статистика:** парний дизайн по камерах; репортити mean APG ± std по сідах і Wilcoxon p-value vs zero-shot та vs найсильнішого бейзлайна.

**Репортинг:** завжди APG (прирости), не абсолютні AP, при порівнянні зі статтею Scenes100.

---

## 6. Ризики та план Б

| Ризик | Ознака | План Б |
|---|---|---|
| Колапс на статичний фон | H_diff падає, |Δbox| до frozen росте, AP падає при зростанні iters | компенсація з flow/frozen-моделі (P4.2), гейт за мінімальним рухом, якір; крива AP vs iters обов'язкова |
| Over-sharpening до хибних країв | w/h боксів систематично дрейфує | entropy floor, EMA-ціль, менша вага TCL, W1 замість KL |
| Сирий KL непридатний (носії не перекриваються) | великі/NaN лосси, багато відкинутих пар | W1 (P4.1), більший reg_scale-запас, менший Δ |
| Сигнал з P3 слабкий | низький Spearman, precision правила ~0.5 | W1, більший Δ з трекінгом, propagated queries (P7.3), фокус на AP75 |
| Відео Scenes100 зникли з YouTube | помилки download | pre-extracted frames зі сховища авторів; dev/full лише з доступних, список у журналі |
| 6 ГБ не вистачає | OOM | D-FINE-N, 512 px, grad-accum, checkpointing декодера, frozen encoder |
| UA-DETRAC недоступний | — | MOT17-static / MOT20 |
| Scenes100 PL-пайплайн не запускається (PyTracking) | — | власний простий IoU/ByteTrack-трекер на детекціях; зафіксувати як відмінність |
| Абсолютні AP непорівнянні зі статтею | — | репортити лише APG; перезапуск їхніх PL на D-FINE |

---

## 7. Журнал виконання

Формат запису:

```
### YYYY-MM-DD — <агент/модель> — <ID задач>
**Зроблено:** ...
**Артефакти:** шляхи до файлів/конфігів/рядків csv
**Проблеми / не вийшло:** ...
**Наступний крок:** ...
```

### 2026-10-03 — Claude Fable 5.1 (Claude Code) — план
**Зроблено:** проаналізовано FORMULA.md, зібрано факти про D-FINE (структура виходів, лосси, DN, чекпоінти), Scenes100 (протокол, класи, маски, референсні числа, PL-пайплайн), UA-DETRAC (недоступний офіційно), DEIM; критичний рев'ю дизайну методу (різні сітки бінів між кадрами → CDF-ресемплінг/W1; колапс на статичний фон → компенсація руху з flow/frozen-моделі; DN-guard, власний локальний лосс поза Hungarian). Створено цей план. Додано вимогу автора: усе середовище в Docker (правило 4a, P0.1a), дані/чекпоінти/результати — лише volumes.
**Артефакти:** `ai-plan.md`.
**Проблеми / не вийшло:** —
**Наступний крок:** P0.1–P0.6.
