"""Region metrics: is the arm, the object and the scene still there and still right.

The tracking metrics score a grid of points grouped after the fact by how far the ground
truth moves them. That grouping is a proxy for "the object", and it cannot see an object
that stops existing: an erased pile of legos is static in the ground truth, so every
point on it lands in the below-4 px bucket and the model is rewarded for keeping it
still. A thing with no trajectory has no trajectory to get wrong.

These modules replace the proxy with a mask. Following WoW-World-Eval (arXiv 2601.04137):
text-prompted detection and segmentation give an arm mask, an object mask and everything
else, then a DINOv2 embedding per region is compared between the real clip and the
generated one.

  * `segment` turns a frame plus text prompts into arm / object / background masks.
  * `propagate` carries a frame-0 mask through the clip.
  * `embed` scores a region: DINOv2 patch features, cosine similarity, per frame.

One asymmetry runs through all of it. Masks come from the *ground truth* and are applied
unchanged to the prediction. Re-segmenting the prediction would let a detector find a
plausible object wherever the model painted one, which erases exactly the signal we are
here to measure: the region the object should occupy, scored on whatever the model put
there instead.
"""

import importlib

__all__ = ['segment', 'propagate', 'embed', 'agree', 'DinoEmbedder']

_LAZY = {'DinoEmbedder': 'embed'}


def __getattr__(name):
    # The submodules pull in transformers and load weights; importing clipeval.regions
    # should not. Same lazy trick as clipeval.tracking.
    if name in _LAZY:
        return getattr(importlib.import_module('.' + _LAZY[name], __name__), name)
    if name in ('segment', 'propagate', 'embed', 'agree'):
        return importlib.import_module('.' + name, __name__)
    raise AttributeError(name)
