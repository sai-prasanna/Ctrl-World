"""CoTracker3 wrapper: frames plus query points in, tracks out.

Kept separate from `metrics.py` so swapping the tracker later touches one file. EWMBench
draws the same line, reading trajectories from a `traj.npy` per episode.

CoTracker is a normal dependency, not vendored. Load the checkpoint from a local path:
`torch.hub.load` reaches GitHub at call time, which fails on a Leonardo compute node,
where there is no network and `HF_HUB_OFFLINE=1` is set. Download `scaled_offline.pth`
on a login node and pass its path.

Offline mode is the default. It sees the whole clip at once and interpolates through
occlusion better than the online mode, and our clips are short enough to fit.
"""

import numpy as np
import torch

CHECKPOINT_URL = ('https://huggingface.co/facebook/cotracker3/resolve/main/'
                  'scaled_offline.pth')


class Tracker:
    """Thin wrapper around CoTrackerPredictor.

    Args:
        checkpoint: local path to scaled_offline.pth (or the online variant).
        device: torch device.
        offline: use the offline predictor; see the module docstring.
    """

    def __init__(self, checkpoint, device='cuda', offline=True):
        from cotracker.predictor import CoTrackerPredictor
        self.checkpoint = checkpoint
        self.offline = offline
        self.device = torch.device(device)
        self.model = CoTrackerPredictor(checkpoint=checkpoint, offline=offline,
                                        v2=False).to(self.device).eval()

    @property
    def info(self):
        return {'tracker': 'cotracker3',
                'mode': 'offline' if self.offline else 'online',
                'checkpoint': self.checkpoint}

    @staticmethod
    def _to_video(frames, device):
        """(T, H, W, 3) uint8 or float in [0, 255] -> (1, T, 3, H, W) float."""
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames)
        return frames.permute(0, 3, 1, 2)[None].float().to(device)

    @torch.no_grad()
    def track(self, frames, queries, backward=False):
        """Track `queries` through `frames`.

        Args:
            frames: (T, H, W, 3) uint8 or float in [0, 255].
            queries: (K, 3) array of (frame_index, x, y) in pixels.
            backward: also track backwards from each query frame.

        Returns:
            tracks: (T, K, 2) float64 pixel coordinates.
            visible: (T, K) bool.
        """
        video = self._to_video(frames, self.device)
        q = torch.as_tensor(np.asarray(queries), dtype=torch.float32,
                            device=self.device)[None]
        tracks, visible = self.model(video, queries=q, backward_tracking=backward)
        return (tracks[0].double().cpu().numpy(),
                visible[0].bool().cpu().numpy())

    def track_pair(self, gt_frames, pred_frames, queries, backward=False):
        """Track the same queries through ground-truth and predicted frames.

        Both sequences must start with the same real conditioning frame, so the two
        tracks begin from identical pixels and any divergence is the model's.
        """
        gt_tracks, gt_vis = self.track(gt_frames, queries, backward=backward)
        pred_tracks, pred_vis = self.track(pred_frames, queries, backward=backward)
        return {'gt_tracks': gt_tracks, 'gt_visible': gt_vis,
                'pred_tracks': pred_tracks, 'pred_visible': pred_vis}


def mark_invisible(tracks, visible, missing=-1.0):
    """Rewrite invisible points as EWMBench's (-1, -1) sentinel."""
    out = np.array(tracks, dtype=np.float64, copy=True)
    out[~np.asarray(visible, dtype=bool)] = missing
    return out
