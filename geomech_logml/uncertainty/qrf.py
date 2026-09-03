"""Quantile Regression Forest (Meinshausen, 2006) on top of sklearn's RandomForest.

For each query row we find, in every tree, the training samples that share its
leaf; pooling those samples across trees yields an empirical predictive
distribution whose quantiles are reported. Intervals are wide where the training
data (in feature space) disagree — i.e. honest epistemic+aleatoric uncertainty.

Complexity note: predictions are chunked; a 300-tree forest on ~700 training rows
evaluates thousands of rows per second, which is ample for well-log curves.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

__all__ = ["QuantileForest"]


class QuantileForest:
    """Thin wrapper exposing `predict_quantiles` on a fitted RandomForestRegressor.

    Parameters
    ----------
    forest : fitted ``RandomForestRegressor``.
    X_train, y_train : the exact training data passed to ``forest.fit`` (sklearn
        forests do not retain ``y``).
    """

    def __init__(self, forest: RandomForestRegressor,
                 X_train: np.ndarray, y_train: np.ndarray) -> None:
        if not isinstance(forest, RandomForestRegressor):
            raise TypeError("QuantileForest wraps a sklearn RandomForestRegressor")
        self.forest = forest
        self._leaf_map: list[dict[int, np.ndarray]] = []
        self._y_train = np.asarray(y_train, dtype=float)
        self._prepare(X_train)

    # ------------------------------------------------------------------
    def _prepare(self, X: np.ndarray) -> None:
        """Pre-compute {leaf_node_id -> train row indices} for every tree (once)."""
        if self._leaf_map:
            return
        leaf_map: list[dict[int, np.ndarray]] = []
        train_leaves = self.forest.apply(X)          # (n_train, n_trees)
        for t in range(train_leaves.shape[1]):
            mapping: dict[int, np.ndarray] = {}
            leaf_ids = train_leaves[:, t]
            for leaf in np.unique(leaf_ids):
                mapping[int(leaf)] = np.where(leaf_ids == leaf)[0]
            leaf_map.append(mapping)
        self._leaf_map = leaf_map

    # ------------------------------------------------------------------
    def predict_quantiles(
        self,
        X_query: np.ndarray,
        quantiles: tuple[float, ...] = (0.05, 0.5, 0.95),
        chunk: int = 512,
    ) -> np.ndarray:
        """Quantile predictions for `X_query`; returns (n_query, n_quantiles)."""
        n = X_query.shape[0]
        out = np.empty((n, len(quantiles)), dtype=float)
        query_leaves = self.forest.apply(X_query)      # (n_query, n_trees)
        y = self._y_train

        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            pooled = []
            for i in range(start, stop):
                idx_parts = [
                    self._leaf_map[t][int(leaf)]
                    for t, leaf in enumerate(query_leaves[i])
                ]
                pooled.append(y[np.concatenate(idx_parts)])
            for j, q in enumerate(quantiles):
                out[start:stop, j] = [np.quantile(r, q) for r in pooled]
        return out

    # ------------------------------------------------------------------
    def predict_intervals(
        self,
        X_query: np.ndarray,
        alpha: float = 0.10,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Symmetric interval (lo, median, hi) at confidence 1 − alpha."""
        lo_q, mid_q, hi_q = alpha / 2.0, 0.5, 1.0 - alpha / 2.0
        q = self.predict_quantiles(X_query, (lo_q, mid_q, hi_q))
        return q[:, 0], q[:, 1], q[:, 2]
