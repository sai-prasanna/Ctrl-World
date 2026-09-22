"""Read the extracted annotation layout as a benchmark episode source.

`extract.py` writes `videos/{split}/{traj}/{view}.mp4` next to
`annotation/{split}/{traj}.json`, and three consumers already read it: Ctrl-World's
training dataset, its clip drawer, and its evaluation rollout. This module adds a fourth
reader, shaped as the `EpisodeSource` that `wmbench` drives a benchmark run through.

Structural, not inherited. `wmbench.core.episode.EpisodeSource` is a runtime-checkable
Protocol, so an object that has the five members satisfies `isinstance` without importing
it; that is what keeps this package at numpy and PyAV and lets it be copied into another
model's checkout. The three small records below (`DatasetProfile`, `SourceProvenance`,
`Episode`) exist for the same reason: they are field-for-field the ones wmbench declares,
duplicated rather than imported, so the dependency points one way only. Their field names
and method semantics are the contract - change one here only together with wmbench.

What this source declares about the data, which the benchmark refuses to guess:

  * The recording is ABC-130k: three cameras, 30 Hz control, a 14-D bimanual joint state.
    None of that changes with the extraction.
  * This copy is 5 Hz and 256x192 with `fov_crop` already applied to the pixels, which is
    `preprocess_id="ctrlworld_fovcrop_v1"`. A crop cannot be undone, so a world model
    declaring any other operating point must not be scored against these frames; the id
    is how the harness catches that instead of reporting the difference as model error.

Latents are deliberately absent. `preprocessing/encode_svd.py` writes `latent_videos/`
for one particular model, and a benchmark that knew about them would be a benchmark for
that model; the Ctrl-World bridge reads them itself.
"""
import glob
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

from .video import read_frames

__all__ = ["ABC_PROFILE", "AnnotationSource", "DatasetProfile", "Episode",
           "SourceProvenance"]

# What one extraction of the release holds. `extract.py` subsamples 30 Hz by rgb_skip=6
# and resizes to 256x192 after the field-of-view crop; the id names that crop so a model
# and a source can be checked against each other.
PREPROCESS_ID = "ctrlworld_fovcrop_v1"
EXTRACTED_FPS = 5.0
EXTRACTED_GEOMETRY = "256x192"


@dataclass(frozen=True)
class DatasetProfile:
    """Recording-level facts: true of the release however it is extracted."""

    name: str
    view_names: tuple
    view_groups: dict = field(default_factory=dict)
    control_hz: float = 0.0
    state_dim: int = 0
    state_layout: tuple = ()

    def to_dict(self):
        return {"name": self.name, "view_names": list(self.view_names),
                "view_groups": {k: list(v) for k, v in self.view_groups.items()},
                "control_hz": self.control_hz, "state_dim": self.state_dim,
                "state_layout": list(self.state_layout)}


@dataclass(frozen=True)
class SourceProvenance:
    """What this copy of the data on disk actually holds."""

    fps: float
    geometry: str = None
    preprocess_id: str = None

    def to_dict(self):
        return {"fps": self.fps, "geometry": self.geometry,
                "preprocess_id": self.preprocess_id}


@dataclass
class Episode:
    """One trajectory: metadata and one state row per stored frame, no pixels."""

    episode_id: str
    task: str = ""
    instruction: str = ""
    fps: float = 0.0
    n_frames: int = 0
    split: str = "val"
    states: np.ndarray = None
    actions: np.ndarray = None
    metadata: dict = field(default_factory=dict)

    @property
    def duration_s(self):
        return self.n_frames / self.fps if self.fps else 0.0

    def frame_indices(self, start_s, count, fps):
        """Stored frame indices for `count` samples at `fps` starting at `start_s`.

        Rounding rather than flooring, so a request at a rate this extraction does not
        store keeps the sampled times centred on the requested ones instead of drifting
        half a frame early.
        """
        if not self.fps:
            raise ValueError(f"episode {self.episode_id} has no frame rate")
        times = start_s + np.arange(count) / float(fps)
        return np.rint(times * self.fps).astype(np.int64)

    def states_at(self, indices):
        if self.states is None:
            raise ValueError(f"episode {self.episode_id} carries no states")
        return np.asarray(self.states)[np.asarray(indices)]


# The Ctrl-World paper reports a third-person row and a wrist row; these are the cameras
# in each. The state order is ABC's own: left arm first, gripper last within each arm,
# which is the order its released policies emit.
ABC_PROFILE = DatasetProfile(
    name="abc130k",
    view_names=("top", "left_wrist", "right_wrist"),
    view_groups={"third_view": ("top",),
                 "wrist_view": ("left_wrist", "right_wrist")},
    control_hz=30.0,
    state_dim=14,
    state_layout=tuple([f"left_j{i}" for i in range(1, 7)] + ["left_gripper"]
                       + [f"right_j{i}" for i in range(1, 7)] + ["right_gripper"]),
)


class AnnotationSource:
    """`EpisodeSource` over `annotation/{split}/*.json` plus `videos/{split}/...`.

    Args:
        root: dataset root, the directory holding `annotation/` and `videos/`. The
            annotation's video paths are relative to it.
        fps, geometry, preprocess_id: provenance of this copy. Defaulted to what
            `extract.py` writes, and overridable so that a re-extraction at another rate
            or resolution is declared rather than assumed.
        annotation_name: the annotation directory, for the alternative index a training
            run sometimes builds beside the canonical one.
        profile: the recording. Overridable only so a re-used layout (the superseded
            DROID extraction has the same shape) can declare its own cameras.
    """

    def __init__(self, root, *, fps=EXTRACTED_FPS, geometry=EXTRACTED_GEOMETRY,
                 preprocess_id=PREPROCESS_ID, annotation_name="annotation",
                 profile=ABC_PROFILE):
        self.root = str(root)
        self.annotation_name = annotation_name
        self.profile = profile
        self.provenance = SourceProvenance(fps=float(fps), geometry=geometry,
                                           preprocess_id=preprocess_id)

    # --------------------------------------------------------------- EpisodeSource

    def list_episodes(self, split="val"):
        """Episode ids, sorted by file name.

        Sorted as strings, not numerically: this is the order `make_eval_clips.py`
        globs in, and the clip list already scored has to stay reproducible from here.
        """
        pattern = f"{self.root}/{self.annotation_name}/{split}/*.json"
        return [os.path.basename(p)[:-len(".json")] for p in sorted(glob.glob(pattern))]

    def load(self, episode_id, split="val"):
        ann = self._annotation(episode_id, split)
        # `texts[0]`, not the `task` key, is the task here: the clip drawer stratifies on
        # this string and writes it into the clip list, so taking the slug instead would
        # redraw a clip list that is already committed and already scored.
        instruction = (ann.get("texts") or [""])[0]
        states = np.asarray(ann["states"], dtype=np.float64)
        actions = ann.get("action")
        return Episode(
            episode_id=str(episode_id),
            task=instruction,
            instruction=instruction,
            fps=float(ann.get("fps", self.provenance.fps)),
            n_frames=int(ann["video_length"]),
            split=split,
            states=states,
            actions=np.asarray(actions, dtype=np.float64) if actions else None,
            metadata={"annotation": ann, "split": split,
                      "task_slug": ann.get("task", ""),
                      "rgb_skip": ann.get("rgb_skip"),
                      "raw_length": ann.get("raw_length")})

    def frames(self, episode, view, t0_s, t1_s):
        """Frames whose timestamps fall in [t0_s, t1_s), as (T, H, W, 3) uint8 RGB.

        Stored frame k covers [k / fps, (k + 1) / fps), so the half-open window takes
        frames floor(t0 * fps) .. ceil(t1 * fps) - 1 and consecutive windows tile without
        overlap. The epsilon is against the float division that produced the seconds:
        a window ending at frame 33 arrives as 6.6000000000000005 s, and a bare ceil
        would read one frame too many.
        """
        fps = float(episode.fps or self.provenance.fps)
        eps = 1e-6
        start = max(0, int(math.floor(t0_s * fps + eps)))
        stop = min(int(episode.n_frames), int(math.ceil(t1_s * fps - eps)))
        if stop <= start:
            return np.empty((0, 0, 0, 3), dtype=np.uint8)
        return read_frames(self._video_path(episode, view), start, stop)

    def intrinsics(self, episode, view):
        """None: this copy has the crop baked in, so intrinsics no longer describe it.

        `mcap_io.fov_crop` consumed them during extraction. A consumer that needs them
        wants the native re-extraction, which records them per episode.
        """
        return None

    # --------------------------------------------------------------------- internals

    def _annotation(self, episode_id, split):
        path = f"{self.root}/{self.annotation_name}/{split}/{episode_id}.json"
        with open(path) as f:
            return json.load(f)

    def _video_path(self, episode, view):
        if view not in self.profile.view_names:
            raise ValueError(f"unknown view {view!r}, expected one of "
                             f"{list(self.profile.view_names)}")
        index = self.profile.view_names.index(view)
        ann = episode.metadata.get("annotation")
        if ann is None:
            ann = self._annotation(episode.episode_id, episode.split)
        return f"{self.root}/{ann['videos'][index]['video_path']}"
