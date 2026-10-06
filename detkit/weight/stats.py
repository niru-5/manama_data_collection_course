"""Statistics for judging weight models (numpy only)."""

from __future__ import annotations

import numpy as np

from .features import SampleTable


def evaluate_model(model, samples: SampleTable) -> dict:
    """Score ``model.predict`` against the true weights: n, mae_g, mape, bias_g (mean pred - true), r2,
    per-crop breakdown (same metrics) and ``rows`` (file, crop, true_g, pred_g, error_g)."""
    rows = []
    for s in samples.with_weight():
        est = model.predict(s.features, s.crop)
        rows.append({"file": s.file, "crop": s.crop, "true_g": s.weight_g, "pred_g": est.weight_g,
                     "error_g": est.weight_g - s.weight_g})
    res = _metrics(rows)
    res["per_crop"] = {c: _metrics([r for r in rows if r["crop"] == c])
                       for c in sorted({r["crop"] or "?" for r in rows})}
    res["rows"] = rows
    return res


def _metrics(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0, "mae_g": None, "mape": None, "bias_g": None, "r2": None}
    err = np.array([r["error_g"] for r in rows])
    true = np.array([r["true_g"] for r in rows])
    nz = true != 0
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return {"n": n, "mae_g": float(np.abs(err).mean()), "bias_g": float(err.mean()),
            "mape": float(np.mean(np.abs(err[nz]) / true[nz])) if nz.any() else None,
            "r2": 1 - float(np.sum(err ** 2)) / ss_tot if n > 1 and ss_tot > 0 else None}


def learning_curve(samples: SampleTable, x_key: str, sizes: list[int], y_key: str = "weight_g",
                   crop: str | None = None, repeats: int = 200):
    """TODO(student) task 3. Return one entry per value in ``sizes``."""
    raise NotImplementedError("learning_curve: student task 3 (docs/STUDENT_TASKS.md)")


def bootstrap_ci(samples: SampleTable, x_key: str, y_key: str = "weight_g", crop: str | None = None,
                 stat: str = "r", n_boot: int = 1000):
    """TODO(student) task 4. ``stat`` is "r" or "slope"; return ``(lo, hi)``."""
    raise NotImplementedError("bootstrap_ci: student task 4 (docs/STUDENT_TASKS.md)")
