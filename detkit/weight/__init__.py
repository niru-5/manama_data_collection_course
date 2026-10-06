"""Weight estimation: detections -> features -> weight (g) -> cost.

The `TODO(student)` markers show what to implement. Tasks are listed in docs/STUDENT_TASKS.md.

    features.py  image_features, Sample, SampleTable   (working)
    models.py    WeightModel, CountTimesConstant (working), LinearAreaModel (TODO)
    stats.py     evaluate_model (working); learning_curve, bootstrap_ci (TODO)
    cost.py      cost_of, propagate (TODO), load_config / save_config
"""

from .cost import cost_of, load_config, propagate, save_config
from .features import Sample, SampleTable, image_features
from .models import (MODELS, CountTimesConstant, Estimate, LinearAreaModel, WeightModel,
                     get_model)
from .stats import bootstrap_ci, evaluate_model, learning_curve

__all__ = ["image_features", "Sample", "SampleTable", "Estimate", "WeightModel", "CountTimesConstant",
           "LinearAreaModel", "MODELS", "get_model",
           "evaluate_model", "learning_curve", "bootstrap_ci", "cost_of", "propagate", "load_config",
           "save_config"]
