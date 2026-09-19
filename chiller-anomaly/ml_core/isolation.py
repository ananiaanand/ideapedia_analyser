"""
ml_core/isolation.py

Isolation Forest anomaly scorer.

Uses multivariate features that capture different anomaly axes:
  - residual:          energy over/under-use vs regression baseline
  - kw_per_ton:        raw efficiency ratio
  - flow_per_load:     flow efficiency proxy
  - regime_distance:   novelty in operating context

Score direction convention:
  IsolationForest.decision_function() returns HIGHER values for NORMAL points.
  We negate it so that HIGHER anomaly_score_if → MORE anomalous.
  Scores are then shifted to [0, 1] via min-max scaling fitted on training data.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ml_core.config import IF_CONTAMINATION, IF_N_ESTIMATORS, IF_RANDOM_STATE, IF_FEATURES

logger = logging.getLogger(__name__)


class ChillerIsolationForest:
    """
    One Isolation Forest model per chiller_id.
    Uses per-equipment per-feature standardization z_i = (x_i - mu) / sigma,
    decision_function d(x), 0-100 health score min-max scaling, and
    contamination=0.05 anomaly thresholding.
    """

    def __init__(
        self,
        contamination: float = IF_CONTAMINATION,
        n_estimators: int = IF_N_ESTIMATORS,
        random_state: int = IF_RANDOM_STATE,
    ) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models: dict[str, IsolationForest] = {}
        self.std_scalers: dict[str, StandardScaler] = {}
        self.d_bounds: dict[str, tuple[float, float]] = {}  # (d_min, d_max) per chiller
        self.scalers: dict[str, MinMaxScaler] = {}          # for [0,1] anomaly score
        self.feature_cols: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "ChillerIsolationForest":
        for chiller_id, group in df.groupby("chiller_id"):
            self._fit_one(chiller_id, group)
        logger.info("ChillerIsolationForest fitted for %d chillers.", len(self.models))
        return self

    def _fit_one(self, chiller_id: str, group: pd.DataFrame) -> None:
        feat_cols = [c for c in IF_FEATURES if c in group.columns]
        if not feat_cols:
            logger.warning("Chiller %s: no IF features available.", chiller_id)
            return

        X = group[feat_cols].dropna()
        if len(X) < 10:
            logger.warning("Chiller %s: too few rows for IF (%d). Skipping.", chiller_id, len(X))
            return

        # 1. Standardize each feature (per equipment, per feature): z_i = (x_i - mu) / sigma
        std_scaler = StandardScaler()
        X_std = std_scaler.fit_transform(X)

        # 2. Fit Isolation Forest
        model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        model.fit(X_std)

        # 3. Decision Function d(x): higher = more normal, lower = more anomalous
        d_scores = model.decision_function(X_std)
        d_min = float(np.min(d_scores))
        d_max = float(np.max(d_scores))

        # Raw anomaly direction (-d(x)): higher = more anomalous
        raw_scores = -d_scores
        scaler = MinMaxScaler()
        scaler.fit(raw_scores.reshape(-1, 1))

        self.models[chiller_id] = model
        self.std_scalers[chiller_id] = std_scaler
        self.d_bounds[chiller_id] = (d_min, d_max)
        self.scalers[chiller_id] = scaler
        self.feature_cols[chiller_id] = feat_cols

        logger.info("Chiller %s: IsolationForest fitted on %d rows (d_min=%.4f, d_max=%.4f).", chiller_id, len(X), d_min, d_max)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 'anomaly_score_if' ∈ [0, 1], 'health_score' ∈ [0, 100], and 'is_anomaly_flag'."""
        result = df.copy()
        result["anomaly_score_if"] = np.nan
        result["health_score"] = np.nan
        result["is_anomaly_flag"] = False

        for chiller_id, group in result.groupby("chiller_id"):
            if chiller_id not in self.models:
                logger.warning("Chiller %s: no IF model found.", chiller_id)
                continue

            model = self.models[chiller_id]
            std_scaler = self.std_scalers[chiller_id]
            d_min, d_max = self.d_bounds.get(chiller_id, (-0.2, 0.2))
            scaler = self.scalers[chiller_id]
            feat_cols = self.feature_cols[chiller_id]

            available = [c for c in feat_cols if c in group.columns]
            X = group[available]
            valid = X.notna().all(axis=1)

            if valid.sum() == 0:
                continue

            X_valid = X[valid]
            X_std = std_scaler.transform(X_valid)

            # Decision Function d(x)
            d_scores = model.decision_function(X_std)
            
            # Anomaly prediction flag (-1 for anomaly, 1 for normal)
            preds = model.predict(X_std)
            is_anomaly = (preds == -1)

            # 3. Health Score = 100 * (((d(x) - d_min) / denom) ** 0.33)
            # A cube-root curve pushes the bulk of normal data to ~95%
            denom = d_max - d_min if d_max > d_min else 1.0
            linear_scale = (d_scores - d_min) / denom
            linear_scale = np.clip(linear_scale, 0.0, 1.0)
            health_scores = 100.0 * (linear_scale ** 0.33)

            # 0-1 Anomaly Score
            raw = -d_scores
            normalised = scaler.transform(raw.reshape(-1, 1)).ravel()
            normalised = np.clip(normalised, 0.0, 1.0)

            result.loc[group.index[valid], "anomaly_score_if"] = normalised
            result.loc[group.index[valid], "health_score"] = np.round(health_scores, 1)
            result.loc[group.index[valid], "is_anomaly_flag"] = is_anomaly

        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: Path | str) -> None:
        path = Path(directory) / "isolation_forest.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
            "models": self.models,
            "std_scalers": self.std_scalers,
            "d_bounds": self.d_bounds,
            "scalers": self.scalers,
            "feature_cols": self.feature_cols,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("ChillerIsolationForest saved to %s.", path)

    @classmethod
    def load(cls, directory: Path | str) -> "ChillerIsolationForest":
        path = Path(directory) / "isolation_forest.pkl"
        if not path.exists():
            raise FileNotFoundError(f"IF model not found at {path}")
        with open(path, "rb") as f:
            payload = pickle.load(f)
        obj = cls(
            contamination=payload["contamination"],
            n_estimators=payload["n_estimators"],
            random_state=payload["random_state"],
        )
        obj.models = payload["models"]
        obj.std_scalers = payload.get("std_scalers", {})
        obj.d_bounds = payload.get("d_bounds", {})
        obj.scalers = payload["scalers"]
        obj.feature_cols = payload["feature_cols"]
        logger.info("ChillerIsolationForest loaded from %s.", path)
        return obj
