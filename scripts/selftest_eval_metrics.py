"""Check the metric machinery in eval_video_metrics.py on synthetic data.

Everything here runs on CPU in a few seconds and needs no checkpoint, so you can verify
the metrics, both distribution-metric backbones, and the aggregation without spending GPU
time. What it does not cover is the rollout itself, which needs a model and a GPU.

    python3 scripts/selftest_eval_metrics.py --i3d_ckpt /path/to/i3d_torchscript.pt
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.eval_video_metrics import (  # noqa: E402
    FeatureBank, anchor_indices, bootstrap_ci, keep_indices, psnr, ssim, stable_seed)

FAILURES = []


def check(name, condition, detail=''):
    print(f'{"PASS" if condition else "FAIL"}  {name}{"  " + detail if detail else ""}')
    if not condition:
        FAILURES.append(name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--i3d_ckpt', type=str, default=None)
    p.add_argument('--skip_backbones', action='store_true')
    args = p.parse_args()

    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    # ---- PSNR -------------------------------------------------------------------
    a = torch.randint(0, 256, (4, 32, 32, 3)).float()
    check('psnr of identical frames is large', bool((psnr(a, a) > 90).all()),
          f'{psnr(a, a)[0]:.1f} dB')
    # a constant offset of d gives a closed form: 20*log10(255/d)
    for d in (1.0, 10.0):
        expected = 20 * np.log10(255.0 / d)
        got = psnr(a, (a - d).clamp(0, 255))
        check(f'psnr matches closed form for offset {d:g}',
              bool(torch.allclose(got, torch.full_like(got, expected), atol=0.3)),
              f'{got.mean():.2f} vs {expected:.2f} dB')

    # ---- SSIM -------------------------------------------------------------------
    s_same = ssim(a, a)
    check('ssim of identical frames is 1', bool(torch.allclose(s_same, torch.ones_like(s_same), atol=1e-4)),
          f'{s_same[0]:.4f}')
    noise = (a + torch.randn_like(a) * 40).clamp(0, 255)
    s_noisy = ssim(a, noise)
    check('ssim drops on noise', bool((s_noisy < s_same).all()), f'{s_noisy.mean():.4f}')

    # ---- frame bookkeeping ------------------------------------------------------
    pred_step, interact_num = 5, 12
    keep, anchor = keep_indices(pred_step, interact_num), anchor_indices(pred_step, interact_num)
    check('kept frame count is 4 per round', len(keep) == 48, f'{len(keep)}')
    check('no frame is scored twice', len(set(keep.tolist())) == len(keep))
    check("each round's frame 0 is excluded", 0 not in keep)
    check('kept frames fit the clip', int(keep.max()) == 48)
    check('anchors precede their targets', bool((anchor < keep).all()))
    check('anchors are round starts', sorted(set(anchor.tolist())) == list(range(0, 48, 4)))

    # ---- seeding ----------------------------------------------------------------
    check('stable_seed is deterministic', stable_seed(0, '12:34') == stable_seed(0, '12:34'))
    check('stable_seed separates clips', stable_seed(0, '12:34') != stable_seed(0, '12:35'))
    check('stable_seed is in range', 0 <= stable_seed(7, 'abc') < 2 ** 31)

    # ---- bootstrap --------------------------------------------------------------
    values = rng.normal(20, 2, size=40)
    episodes = [f'ep{i // 2}' for i in range(40)]  # two clips per episode
    mean, lo, hi = bootstrap_ci(values, episodes, 500, 0)
    check('bootstrap mean matches', abs(mean - values.mean()) < 1e-9, f'{mean:.4f}')
    check('bootstrap interval brackets the mean', lo < mean < hi, f'[{lo:.3f}, {hi:.3f}]')
    _, lo1, hi1 = bootstrap_ci(values, ['one'] * 40, 200, 0)
    check('a single episode gives a zero-width interval', abs(hi1 - lo1) < 1e-9)

    # ---- FID and FVD backbones --------------------------------------------------
    if args.skip_backbones:
        print('SKIP  backbones')
    else:
        bank = FeatureBank(torch.device('cpu'), torch.float32,
                           fid=True, i3d_ckpt=args.i3d_ckpt)
        frames = torch.randint(0, 256, (6, 64, 64, 3)).float()
        bank.add_frames('v/pred', frames)
        bank.add_frames('v/real', frames)
        videos = torch.randint(0, 256, (2, 16, 64, 64, 3)).float()
        bank.add_videos('v/pred', videos)
        bank.add_videos('v/real', videos)
        out = bank.results(['v'])
        check('fid backbone ran', 'fid' in out, str(bank.backbones.get('fid', '')))
        # Identical feature sets must give a Frechet distance of zero.
        if 'fid' in out:
            check('fid of a set against itself is 0', abs(out['fid']['v']) < 1e-3,
                  f'{out["fid"]["v"]:.6f}')
        if args.i3d_ckpt:
            check('fvd backbone ran', 'fvd' in out, str(bank.backbones.get('fvd', '')))
            if 'fvd' in out:
                check('fvd of a set against itself is 0', abs(out['fvd']['v']) < 1e-3,
                      f'{out["fvd"]["v"]:.6f}')
        else:
            print('SKIP  fvd (no --i3d_ckpt)')

    print()
    if FAILURES:
        print(f'{len(FAILURES)} check(s) failed: {", ".join(FAILURES)}')
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
