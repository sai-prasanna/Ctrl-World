"""Tracking-based metrics: does the arm, the scene and the object behave correctly.

Pixel metrics score appearance rather than control. PSNR and SSIM reward a blurred mean
future, and FID and FVD cannot tell whether an object ended up in the right place. These
modules track points through both videos and compare where they went.

  * `seeding` picks which points to track.
  * `extract` runs CoTracker3 over frames and returns tracks.
  * `metrics` scores tracks against tracks.

The split follows EWMBench, which reads trajectories from a file per episode; keeping it
means swapping the tracker later touches only `extract`.
"""

import importlib

from . import metrics, seeding  # noqa: F401

__all__ = ['metrics', 'seeding', 'extract', 'Tracker']


def __getattr__(name):
    # `extract` imports cotracker, which is not needed to use the rest of the package.
    # importlib rather than `from . import extract`: the latter looks the attribute up on
    # this module first, which lands back in here and recurses.
    if name in ('extract', 'Tracker'):
        extract = importlib.import_module('.extract', __name__)
        return extract if name == 'extract' else extract.Tracker
    raise AttributeError(name)
