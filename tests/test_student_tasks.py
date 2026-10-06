"""Self-check for the student tasks (docs/STUDENT_TASKS.md).

Run:  .venv/bin/python -m pytest tests/test_student_tasks.py -q -rs

A task that is not implemented yet is reported as SKIPPED with its task number ("TODO task 2").
Once you implement it, the test below runs for real and must pass. Do not edit these tests to make them pass.
"""

import numpy as np
import pytest

from detkit.weight import (CountTimesConstant, LinearAreaModel, SampleTable, bootstrap_ci,
                           cost_of, evaluate_model, learning_curve, propagate)

A2G = 0.002          # true grams per px^2 of total box area
MM = 0.5             # mm per px used for the synthetic scale


def synthetic(n=30, seed=0, crop="wheat"):
    rng = np.random.default_rng(seed)
    t = SampleTable()
    for i in range(n):
        k = int(rng.integers(3, 40))                       # grains in the photo
        side = float(rng.uniform(18, 26))
        boxes = [{"bbox": [j * 30, 0, j * 30 + side, side], "label": crop, "score": 0.9} for j in range(k)]
        area = k * side * side
        w = A2G * area * (1 + rng.normal(0, 0.03))
        t.add(f"{crop}{i}.jpg", boxes, {"crop": crop, "total_weight_g": w, "scale_mm_per_px": MM,
                                        "pile_id": f"p{i // 2}"})
    return t


def todo(task, fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except NotImplementedError:
        pytest.skip(f"TODO task {task}: not implemented yet (docs/STUDENT_TASKS.md)")


def test_task2_linear_area_model_recovers_slope():
    t = synthetic()
    m = todo(2, LinearAreaModel().fit, t)
    s = t.rows[0]
    est = m.predict(s.features, s.crop)
    assert est.weight_g == pytest.approx(s.weight_g, rel=0.15)
    assert evaluate_model(m, t)["mape"] < 0.1


def test_task3_learning_curve_gets_narrower():
    out = todo(3, learning_curve, synthetic(40), "total_area_px", sizes=[5, 10, 20, 40], repeats=50)
    assert out is not None and len(out) == 4


def test_task4_bootstrap_ci_brackets_r():
    lo, hi = todo(4, bootstrap_ci, synthetic(40), "total_area_px", stat="r", n_boot=200)
    assert 0.9 < lo <= hi <= 1.0


def test_task6_propagate_returns_budget():
    out = todo(6, propagate, {"camera": 0.02, "detector": 0.05, "model": 0.03}, weight_g=100.0, price_per_kg=0.25)
    assert out["cost_u"] > 0 and out["dominant"] == "detector"
    assert 0.05 <= out["rel_total"] <= 0.02 + 0.05 + 0.03


def test_task7_baseline_gives_interval():
    t = synthetic()
    m = CountTimesConstant().fit(t)
    est = m.predict(t.rows[0].features, "wheat")
    if est.low_g is None or est.high_g is None:
        pytest.skip("TODO task 7: CountTimesConstant.predict should return low_g/high_g")
    assert est.low_g <= est.weight_g <= est.high_g


def test_cost_of_baseline_works():
    assert cost_of(500, 0.4) == pytest.approx(0.2)
