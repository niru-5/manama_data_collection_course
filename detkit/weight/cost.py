"""Weight -> cost, and the per-workdir config (``W/weight_config.json``)."""

from __future__ import annotations

import json
from pathlib import Path

CONFIG_NAME = "weight_config.json"

# grams per kernel and price per kg. Starting values only: replace them with your own (task 8).
DEFAULT_CONFIG: dict = {
    "model": "count_x_constant",
    "constants": {"default": 0.04, "wheat": 0.04, "sunflower": 0.05},
    "price_per_kg": {"default": 0.25},                                  # currency per kg
}


def load_config(workdir) -> dict:
    """Config of a workdir; missing file (or missing keys) -> the defaults above."""
    p = Path(workdir) / CONFIG_NAME
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if p.exists():
        saved = json.loads(p.read_text(encoding="utf-8"))
        for k, v in saved.items():
            cfg[k] = {**cfg[k], **v} if isinstance(v, dict) and isinstance(cfg.get(k), dict) else v
    return cfg


def save_config(workdir, cfg: dict) -> Path:
    p = Path(workdir) / CONFIG_NAME
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return p


def cost_of(weight_g: float, price_per_kg: float) -> float:
    """Cost of ``weight_g`` grams at ``price_per_kg`` per kilogram."""
    return weight_g / 1000.0 * price_per_kg


def propagate(rel_errors: dict[str, float], weight_g: float, price_per_kg: float) -> dict:
    """TODO(student) task 6. ``rel_errors`` = {stage: relative error}.

    Return a dict with keys ``rel_total``, ``cost``, ``cost_u``, ``dominant`` (stage name), ``budget`` ({stage: share}).
    """
    raise NotImplementedError("propagate(): student task 6, see docs/STUDENT_TASKS.md")
