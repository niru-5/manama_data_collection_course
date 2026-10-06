# Student tasks

Find them with `git grep -n "TODO(student)" -- detkit`. Self-check: `.venv/bin/python -m pytest tests/test_student_tasks.py -q -rs`
(skipped tests = tasks not done yet; passing tests only check the interface, not whether your approach is sound).
Statistics need at least 3 weighed photos per crop, otherwise they return `None`.
All tasks use the provided baseline detector; none of them requires retraining it.

How to solve each task is up to you. Justify your choices in your report.

| # | Where | Deliver |
|---|---|---|
| 1 | (your own analysis) | Which detection feature relates best to weight, with evidence |
| 2 | `detkit/weight/models.py` `LinearAreaModel` | A working model |
| 3 | `detkit/weight/stats.py` `learning_curve` | One entry per size; how many samples you need, and why |
| 4 | `detkit/weight/stats.py` `bootstrap_ci` | `(lo, hi)` |
| 5 | `detkit/weight/stats.py` `evaluate_model` | A fair comparison of all models |
| 6 | `detkit/weight/cost.py` `propagate` | Dict with `rel_total`, `cost`, `cost_u`, `dominant`, `budget`; your uncertainty budget |
| 7 | `detkit/weight/models.py` `CountTimesConstant` | An improved baseline that also returns `low_g` / `high_g` |
| 8 | `detkit/weight/cost.py` `DEFAULT_CONFIG`, `detkit weight calibrate` | Your constants and prices, and how a new crop is added |
| 9 | (research, report) | Industry metrics for grain, and your result expressed in them |
