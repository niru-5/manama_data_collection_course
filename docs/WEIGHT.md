# Weight estimation (`detkit/weight/`, `detkit weight ...`)

Photo -> detections -> features -> weight -> cost. Baseline: kernel count x a constant per crop.

## Files
| file | content | status |
|---|---|---|
| `features.py` | `image_features`, `Sample`, `SampleTable` | works |
| `models.py` | `CountTimesConstant`, `LinearAreaModel`, `MODELS`, `get_model` | baseline works; `LinearAreaModel` TODO |
| `stats.py` | `evaluate_model`; `learning_curve`, `bootstrap_ci` | last two TODO |
| `cost.py` | `cost_of`, `load_config`/`save_config`, `propagate` | `propagate` TODO |

Tasks: [STUDENT_TASKS.md](STUDENT_TASKS.md). Per-workdir config: `<workdir>/weight_config.json`
(model, grams per kernel per crop, price per kg).

## CLI
```bash
python -m detkit weight predict   --workdir $W (--predictions P | --use reviewed) [--model NAME] [--reviewed-only] [--out CSV]
python -m detkit weight calibrate --workdir $W        # fits count_x_constant per crop, saves weight_config.json
```
`predict` prints per-photo count, estimated weight and error vs `total_weight_g`; models still TODO exit with code 2.

## Python
```python
from detkit.project import Project
from detkit.weight import SampleTable, CountTimesConstant, evaluate_model
t = SampleTable.from_project(Project.load("runs/mine")).with_weight()
print(evaluate_model(CountTimesConstant().fit(t), t)["mape"])
```
