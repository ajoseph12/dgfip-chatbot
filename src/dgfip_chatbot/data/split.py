"""Stratified dev/test split of the eval questions (Phase 1)."""

import random

import pandas as pd


def stratified_split(
    df: pd.DataFrame,
    *,
    test_size: float = 0.30,
    seed: int = 42,
    stratify_by: str = "fiche_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into dev/test, stratified so each ``stratify_by`` group keeps the
    ~``test_size`` proportion and appears in *both* splits. Deterministic given ``seed``."""
    # Seeded RNG → same split every run (reproducibility).
    rnd = random.Random(seed)
    dev_idx: list[int] = []
    test_idx: list[int] = []
    # Split within each group (e.g. each fiche) separately, so the proportion holds per
    # group and no fiche ends up entirely in one side. sort=True makes iteration order
    # deterministic.
    for _, group in df.groupby(stratify_by, sort=True):
        idx = list(group.index)
        rnd.shuffle(idx)
        # ~test_size of the group to test, but clamp to [1, len-1] so BOTH splits get >=1.
        n_test = min(max(1, round(len(idx) * test_size)), len(idx) - 1)
        test_idx.extend(idx[:n_test])
        dev_idx.extend(idx[n_test:])
    # Sort indices to restore original row order, tag the split, and renumber the index.
    dev = df.loc[sorted(dev_idx)].assign(split="dev").reset_index(drop=True)
    test = df.loc[sorted(test_idx)].assign(split="test").reset_index(drop=True)
    return dev, test
