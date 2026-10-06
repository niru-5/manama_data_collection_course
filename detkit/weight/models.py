"""Weight models: features -> Estimate. ``CountTimesConstant`` works; LinearAreaModel is a student task."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from .cost import DEFAULT_CONFIG
from .features import SampleTable


@dataclass
class Estimate:
    weight_g: float
    low_g: float | None = None       # lower/upper bound of an interval, if the model can give one
    high_g: float | None = None
    note: str = ""


class WeightModel:
    """Interface. Subclass, set ``name``/``description``, implement ``fit`` and ``predict``.

    ``params()``/``set_params()`` hold everything needed to save and restore a fitted model as JSON.
    """
    name = "base"
    description = ""
    implemented = False

    def fit(self, samples: SampleTable) -> "WeightModel":
        raise NotImplementedError

    def predict(self, features: dict, crop: str | None = None) -> Estimate:
        raise NotImplementedError

    def params(self) -> dict:
        return {}

    def set_params(self, params: dict) -> None:
        pass

    def to_dict(self) -> dict:
        return {"name": self.name, "params": self.params()}

    @staticmethod
    def from_dict(d: dict) -> "WeightModel":
        m = get_model(d["name"])
        m.set_params(d.get("params", {}))
        return m

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "WeightModel":
        return WeightModel.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class CountTimesConstant(WeightModel):
    """Baseline: weight = number of detected kernels x grams-per-kernel of the crop.

    ``constants`` = {"default": g, "<crop>": g}.
    """
    name = "count_x_constant"
    description = "weight = kernel count x constant g/kernel per crop (baseline)"
    implemented = True

    def __init__(self, constants: dict[str, float] | None = None):
        self.constants = dict(constants or DEFAULT_CONFIG["constants"])

    def fit(self, samples: SampleTable) -> "CountTimesConstant":
        """Calibrate: per crop, constant = mean(weight / count) over weighed photos.
        Crops without data keep their current constant. ``default`` = pooled mean over all crops."""
        ratios: dict[str, list[float]] = {}
        for s in samples.with_weight():
            if s.features.get("count"):
                ratios.setdefault(s.crop or "default", []).append(s.weight_g / s.features["count"])
        for crop, r in ratios.items():
            self.constants[crop] = statistics.fmean(r)
        allr = [x for r in ratios.values() for x in r]
        if allr:
            self.constants["default"] = statistics.fmean(allr)
        # TODO(student) task 7
        return self

    def predict(self, features: dict, crop: str | None = None) -> Estimate:
        known = crop in self.constants
        g = self.constants[crop] if known else self.constants["default"]
        note = "" if known or crop is None else f"no constant for crop {crop!r}, used default"
        # TODO(student) task 7: low_g / high_g
        return Estimate(features.get("count", 0) * g, note=note)

    def params(self) -> dict:
        return {"constants": self.constants}

    def set_params(self, params: dict) -> None:
        self.constants = dict(params["constants"])


class LinearAreaModel(WeightModel):
    """TODO(student) task 2: weight from pixel area.

    ``fit(samples)`` returns self; ``predict(features, crop)`` returns an ``Estimate``.
    """
    name = "linear_area"
    description = "weight from pixel area (student task 2)"

    def fit(self, samples: SampleTable) -> "LinearAreaModel":
        raise NotImplementedError("LinearAreaModel.fit: student task 2 (docs/STUDENT_TASKS.md).")

    def predict(self, features: dict, crop: str | None = None) -> Estimate:
        raise NotImplementedError("LinearAreaModel.predict: student task 2 (docs/STUDENT_TASKS.md).")


MODELS: dict[str, type[WeightModel]] = {
    m.name: m for m in (CountTimesConstant, LinearAreaModel)
}


def get_model(name: str, **kw) -> WeightModel:
    if name not in MODELS:
        raise KeyError(f"unknown weight model {name!r}; available: {sorted(MODELS)}")
    return MODELS[name](**kw)
