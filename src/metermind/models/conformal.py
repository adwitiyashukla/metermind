from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MIN_GROUP_CALIBRATION = 50


def conformity_scores(y_true, lo, hi) -> np.ndarray:
    y = np.asarray(y_true, dtype=float)
    low = np.asarray(lo, dtype=float)
    high = np.asarray(hi, dtype=float)
    return np.maximum(low - y, y - high)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = scores.size
    if n == 0:
        return float("nan")
    level = np.ceil((n + 1) * (1 - alpha)) / n
    if level > 1.0:
        return float(np.max(scores))
    return float(np.quantile(scores, level, method="higher"))


@dataclass
class ConformalCalibration:
    alpha: float
    q_hat: float
    n_calibration: int
    group_q_hat: dict[str, float] = field(default_factory=dict)
    group_counts: dict[str, int] = field(default_factory=dict)

    @property
    def target_coverage(self) -> float:
        return 1.0 - self.alpha

    @property
    def is_grouped(self) -> bool:
        return bool(self.group_q_hat)

    @classmethod
    def fit(
        cls,
        y_true,
        lo,
        hi,
        alpha: float = 0.2,
        groups: pd.Series | np.ndarray | None = None,
    ) -> ConformalCalibration:
        scores = conformity_scores(y_true, lo, hi)
        finite = np.isfinite(scores)
        global_q = conformal_quantile(scores[finite], alpha)

        group_q: dict[str, float] = {}
        group_n: dict[str, int] = {}
        if groups is not None:
            labels = pd.Series(np.asarray(groups)).astype(str).to_numpy()
            for label in pd.unique(labels[finite]):
                mask = finite & (labels == label)
                n = int(mask.sum())
                group_n[label] = n
                if n < MIN_GROUP_CALIBRATION:
                    logger.warning(
                        "  group %-14s only %d calibration point(s); using the global q_hat",
                        label, n,
                    )
                    group_q[label] = global_q
                else:
                    group_q[label] = conformal_quantile(scores[mask], alpha)

        return cls(
            alpha=float(alpha),
            q_hat=global_q,
            n_calibration=int(finite.sum()),
            group_q_hat=group_q,
            group_counts=group_n,
        )

    def apply(
        self, lo, hi, groups: pd.Series | np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        low = np.asarray(lo, dtype=float)
        high = np.asarray(hi, dtype=float)

        if self.is_grouped and groups is not None:
            labels = pd.Series(np.asarray(groups)).astype(str).to_numpy()
            widen = np.array([self.group_q_hat.get(g, self.q_hat) for g in labels], dtype=float)
        else:
            widen = np.full(low.shape, self.q_hat, dtype=float)

        return low - widen, high + widen

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "target_coverage": self.target_coverage,
            "q_hat": self.q_hat,
            "n_calibration": self.n_calibration,
            "group_q_hat": self.group_q_hat,
            "group_counts": self.group_counts,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> ConformalCalibration:
        return cls(
            alpha=payload["alpha"],
            q_hat=payload["q_hat"],
            n_calibration=payload["n_calibration"],
            group_q_hat=payload.get("group_q_hat", {}),
            group_counts=payload.get("group_counts", {}),
        )

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: Path) -> ConformalCalibration:
        return cls.from_dict(json.loads(Path(path).read_text()))


def empirical_coverage(y_true, lo, hi) -> float:
    y = np.asarray(y_true, dtype=float)
    low = np.asarray(lo, dtype=float)
    high = np.asarray(hi, dtype=float)
    mask = np.isfinite(y) & np.isfinite(low) & np.isfinite(high)
    if not mask.any():
        return float("nan")
    return float(np.mean((y[mask] >= low[mask]) & (y[mask] <= high[mask])) * 100)


def mean_interval_width(lo, hi) -> float:
    low = np.asarray(lo, dtype=float)
    high = np.asarray(hi, dtype=float)
    mask = np.isfinite(low) & np.isfinite(high)
    return float(np.mean(high[mask] - low[mask])) if mask.any() else float("nan")


def coverage_report(
    y_true,
    lo,
    hi,
    groups: pd.Series | np.ndarray | None = None,
    target: float = 80.0,
) -> pd.DataFrame:
    rows = [{
        "group": "ALL",
        "n": int(np.isfinite(np.asarray(y_true, dtype=float)).sum()),
        "coverage_pct": round(empirical_coverage(y_true, lo, hi), 2),
        "target_pct": target,
        "mean_width": round(mean_interval_width(lo, hi), 3),
    }]

    if groups is not None:
        labels = pd.Series(np.asarray(groups)).astype(str).to_numpy()
        y = np.asarray(y_true, dtype=float)
        low = np.asarray(lo, dtype=float)
        high = np.asarray(hi, dtype=float)
        for label in sorted(pd.unique(labels)):
            mask = labels == label
            rows.append({
                "group": label,
                "n": int(mask.sum()),
                "coverage_pct": round(empirical_coverage(y[mask], low[mask], high[mask]), 2),
                "target_pct": target,
                "mean_width": round(mean_interval_width(low[mask], high[mask]), 3),
            })

    frame = pd.DataFrame(rows)
    frame["coverage_gap_pp"] = (frame["coverage_pct"] - frame["target_pct"]).round(2)
    return frame
