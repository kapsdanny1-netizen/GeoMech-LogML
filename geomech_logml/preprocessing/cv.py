"""Well-wise (spatially independent) cross-validation splitters.

Hard rule from the literature review: **never** split rock-physics ML data by random
row sampling — depth-adjacent samples are strongly autocorrelated, which leaks and
inflates scores. Every splitter here keeps a well entirely inside a single fold.

Both splitters implement the scikit-learn ``split(X, y=None, groups=well_ids)``
interface so they can be used with ``cross_val_predict(..., groups=...)`` or our own
loops.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold

from geomech_logml.config import GLOBAL_SEED

__all__ = ["WellKFold", "LeaveOneWellOut", "make_splitter"]


class WellKFold:
    """K-fold CV where folds are *groups of wells*, never row mixtures.

    Parameters
    ----------
    n_splits : number of well-groups (default 5, or fewer if fewer wells exist).
    seed : RNG seed for the well shuffle.
    """

    def __init__(self, n_splits: int = 5, seed: int = GLOBAL_SEED) -> None:
        self.n_splits = max(2, int(n_splits))
        self.seed = seed
        self._kfold = KFold(n_splits=self.n_splits, shuffle=True,
                            random_state=seed)

    # -- sklearn-compatible interface ---------------------------------------
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        if groups is None:
            raise ValueError("WellKFold requires `groups` (well ids).")
        return min(self.n_splits, len(np.unique(groups)))

    def split(self, X=None, y=None, groups=None):
        """Yield (train_row_idx, test_row_idx) arrays; `groups` = well id per row."""
        if groups is None:
            raise ValueError("WellKFold requires `groups` (well ids).")
        wells = np.asarray(groups)
        unique = np.array(sorted(np.unique(wells)))
        if len(unique) < 2:
            raise ValueError("Well-wise CV needs at least 2 wells.")
        rng = np.random.default_rng(self.seed)
        unique = unique.copy()
        rng.shuffle(unique)

        # Use sklearn's KFold over *well names* to build balanced well groups.
        n_eff = min(self.n_splits, len(unique))
        kf = KFold(n_splits=n_eff, shuffle=True, random_state=self.seed)
        well_to_fold = {}
        for fold_id, (_, test_idx) in enumerate(kf.split(unique.reshape(-1, 1))):
            for wi in test_idx:
                well_to_fold[unique[wi]] = fold_id

        fold_ids = np.array([well_to_fold[w] for w in wells])
        for fold in range(n_eff):
            train_idx = np.where(fold_ids != fold)[0]
            test_idx = np.where(fold_ids == fold)[0]
            if train_idx.size and test_idx.size:
                yield train_idx, test_idx

    @property
    def name(self) -> str:
        return f"WellKFold(k={self.n_splits})"


class LeaveOneWellOut:
    """Leave-one-well-out CV: the strictest spatial test for small datasets."""

    def __init__(self, seed: int = GLOBAL_SEED) -> None:
        self.seed = seed

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        if groups is None:
            raise ValueError("LeaveOneWellOut requires `groups` (well ids).")
        return len(np.unique(groups))

    def split(self, X=None, y=None, groups=None):
        if groups is None:
            raise ValueError("LeaveOneWellOut requires `groups` (well ids).")
        wells = np.asarray(groups)
        for w in sorted(np.unique(wells)):
            test_idx = np.where(wells == w)[0]
            train_idx = np.where(wells != w)[0]
            if train_idx.size:
                yield train_idx, test_idx

    @property
    def name(self) -> str:
        return "LeaveOneWellOut"


def make_splitter(strategy: str = "well_kfold", n_wells: int = 99,
                  n_splits: int = 5, seed: int = GLOBAL_SEED):
    """Factory: choose a well-wise splitter appropriate for the number of wells."""
    if strategy == "leave_one_well_out":
        return LeaveOneWellOut(seed=seed)
    k = min(n_splits, n_wells)
    if k < 2:
        raise ValueError("Need at least 2 wells for well-wise cross-validation.")
    return WellKFold(n_splits=k, seed=seed)
