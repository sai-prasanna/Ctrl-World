"""Aggregation helpers shared by every metric family."""

from collections import defaultdict

import numpy as np


def bootstrap_ci(values, episode_ids, n_boot=1000, seed=0):
    """Percentile CI resampling episodes, not clips: clips from one episode are not
    independent, so resampling clips would understate the interval."""
    values = np.asarray(values, dtype=np.float64)
    by_episode = defaultdict(list)
    for value, episode in zip(values, episode_ids):
        by_episode[episode].append(value)
    keys = list(by_episode)
    means = {k: float(np.mean(v)) for k, v in by_episode.items()}
    rng = np.random.default_rng(seed)
    draws = [float(np.mean([means[keys[i]] for i in rng.integers(0, len(keys), len(keys))]))
             for _ in range(n_boot)]
    return (float(np.mean(values)), float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)))
