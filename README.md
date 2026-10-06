# Course project: estimate the value of grain from a photo

> Photo of grain  ->  count / pixels  ->  weight (g)  ->  cost

You get a working grain detector (RF-DETR) and a toolkit (`detkit`) to collect photos and turn detections into
weight and price. The course is about **designing the data-collection process** so that the provided model can be
used to estimate weight: plan it, identify the sources of error, and measure them where you can. You are graded on
how you **collect data, measure uncertainty and justify your conclusions**, not on detector accuracy.

Retraining the detector is possible (`detkit train`) but optional and not part of the course.

## Links
- Code: https://github.com/niru-5/manama_data_collection_course
- Dataset: https://huggingface.co/datasets/niru-5/manama_course_dataset_grain_detection
- Baseline model: https://huggingface.co/niru-5/manama_course_rfdetr_grain_detection

## Linux/macOS vs Windows
Commands are given twice: **Linux/macOS** (bash) and **Windows** (PowerShell). The differences:
* Path separators: Linux/macOS use `/`, Windows uses `\` (`runs/mine` vs `runs\mine`).
* The virtual environment: `.venv/bin/python` on Linux/macOS, `.venv\Scripts\python` on Windows.
* A long command continues on the next line with `\` (bash) or `` ` `` (PowerShell).

The docs in `docs/` show the Linux/macOS form only; translate it the same way on Windows. `$D` there stands for
`.venv/bin/python -m detkit` (Windows: `.venv\Scripts\python -m detkit`), `$W` for your workdir (e.g. `runs/mine`).
Alternative on Windows: use WSL2 (Ubuntu) and follow the Linux commands as written.

## Quick start

**Linux/macOS**
```bash
# 1. install (ONE of cpu / gpu)
curl -LsSf https://astral.sh/uv/install.sh | sh                 # uv, once
uv sync --extra cpu --extra app --extra dev                      # or: --extra gpu
.venv/bin/python -m detkit doctor
.venv/bin/python -m pytest -q                                    # environment check (offline)

# 2. project folder (one workdir = one batch of photos)
.venv/bin/python -m detkit init --workdir runs/mine --classes corn,peanuts,popcorn_corn,pumpkin,sunflower,wheat \
    --flow cpu --tile 1500 --proposer-ckpt weights/grain_rfdetr_small_v1/final    # --flow cpu | gpu | hf

# 3. add photos, then open the app (http://127.0.0.1:7860)
.venv/bin/python -m detkit import-photos --workdir runs/mine --images my_photos/ --crop wheat
.venv/bin/python -m detkit app --workdir runs/mine --ckpt weights/grain_rfdetr_small_v1/final
```

**Windows (PowerShell)**
```powershell
# 1. install (ONE of cpu / gpu; gpu needs an NVIDIA driver)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # uv, once
uv sync --extra cpu --extra app --extra dev                      # or: --extra gpu
.venv\Scripts\python -m detkit doctor
.venv\Scripts\python -m pytest -q                                # environment check (offline)

# 2. project folder (one workdir = one batch of photos)
.venv\Scripts\python -m detkit init --workdir runs\mine --classes corn,peanuts,popcorn_corn,pumpkin,sunflower,wheat `
    --flow cpu --tile 1500 --proposer-ckpt weights\grain_rfdetr_small_v1\final    # --flow cpu | gpu | hf

# 3. add photos, then open the app (http://127.0.0.1:7860)
.venv\Scripts\python -m detkit import-photos --workdir runs\mine --images my_photos\ --crop wheat
.venv\Scripts\python -m detkit app --workdir runs\mine --ckpt weights\grain_rfdetr_small_v1\final
```
A Hugging Face token (`HF_TOKEN=...` in `.env`) is only needed for pushing a model to the Hub (optional). Never commit it.

## Data and baseline model (Hugging Face Hub, public: no token needed)
Dataset: 6 classes, COCO format; read its README. Then try the baseline on a test photo
(score 0.38 = best-F1 threshold chosen on validation).

**Linux/macOS**
```bash
.venv/bin/hf download niru-5/manama_course_dataset_grain_detection --repo-type dataset --local-dir ../data/grain_detection_public
.venv/bin/hf download niru-5/manama_course_rfdetr_grain_detection --local-dir weights/grain_rfdetr_small_v1/final

.venv/bin/python -m detkit predict --model weights/grain_rfdetr_small_v1/final \
    --images ../data/grain_detection_public/photos/test/corn_0007.jpg \
    --tile 1500 --overlap 0.2 --score 0.38 --out runs/first_predictions
```

**Windows (PowerShell)**
```powershell
.venv\Scripts\hf download niru-5/manama_course_dataset_grain_detection --repo-type dataset --local-dir ..\data\grain_detection_public
.venv\Scripts\hf download niru-5/manama_course_rfdetr_grain_detection --local-dir weights\grain_rfdetr_small_v1\final

.venv\Scripts\python -m detkit predict --model weights\grain_rfdetr_small_v1\final `
    --images ..\data\grain_detection_public\photos\test\corn_0007.jpg `
    --tile 1500 --overlap 0.2 --score 0.38 --out runs\first_predictions
```

Optional, not part of the course: to train on the dataset, create a workdir and link the dataset folders into it
(same for `val`); `detkit train` reads `<workdir>/coco/{train,val}/labels.json`.
* Linux/macOS: `ln -s $(realpath ../data/grain_detection_public/coco/train) runs/mine/coco/train`
* Windows: `New-Item -ItemType Junction -Path runs\mine\coco\train -Target (Resolve-Path ..\data\grain_detection_public\coco\train)`
  (a junction needs no admin rights), or copy the folder.

How the baseline was trained, with commands, settings and the scores to expect: `weights/grain_rfdetr_small_v1/TRAINING.md`.
Baseline (photo level): F1 0.986 on val, 0.972 on test, measured against the dataset's automatic (unreviewed) labels.
Licenses: the code in this repository is MIT (`LICENSE`); the dataset and the model are CC BY-NC 4.0 (education, non-commercial).

## Workflow

| Step | Command / place | Doc |
|---|---|---|
| Plan + collect photos and weights | phone + balance | [docs/DATA_COLLECTION.md](docs/DATA_COLLECTION.md) |
| Check detections, enter metadata | `detkit app`, tab **Review** | [docs/APP.md](docs/APP.md) |
| (optional) Split + train | `detkit split --from-reviewed`, `detkit train` | [docs/TRAINING.md](docs/TRAINING.md) |
| Evaluate the detector | `detkit eval` | [docs/EVAL.md](docs/EVAL.md) |
| Count -> weight -> cost | app tab **Inference**, `detkit weight ...` | [docs/WEIGHT.md](docs/WEIGHT.md) |
| Your tasks | `TODO(student)` markers | [docs/STUDENT_TASKS.md](docs/STUDENT_TASKS.md) |

Find your tasks in the code: `git grep -n "TODO(student)" -- detkit`. Task table: [docs/STUDENT_TASKS.md](docs/STUDENT_TASKS.md).
Check your work: `.venv/bin/python -m pytest tests/test_student_tasks.py -q -rs`
(Windows: `.venv\Scripts\python -m pytest tests\test_student_tasks.py -q -rs`).

## Folder map
```
detkit/app/       Gradio app (Review tab: ui.py, Inference tab: inference_ui.py)
detkit/weight/    features, models*, stats*, cost*   (* contains TODOs)
detkit/           train.py predict.py evaluate.py split.py tiling.py store.py schema.py
detkit/vendor/    copied trainer (leave alone)
docs/             the documents above
weights/grain_rfdetr_small_v1/ baseline model (README.md = model card, TRAINING.md = how it was trained)
tests/            offline tests
runs/             YOUR data and checkpoints (git-ignored)
```
`detkit/schema.py` defines the metadata fields (app form + tables).

## Rules
1. **Split by photo, never by tile** (`detkit split` does it).
2. **One pile = one sample.** Photos of the same pile share a `pile_id`.
3. **Only saved (reviewed) photos count**, for the tables and for (optional) training.
4. **Use the best-F1 threshold** from `detkit eval`, not 0.5.
5. **State what your metrics are measured against** (reviewed boxes or pseudo-labels).
6. Keep every checkpoint you report from. Never paste your `HF_TOKEN`.


