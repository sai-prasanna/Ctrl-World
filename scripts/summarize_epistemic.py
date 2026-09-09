"""Summarize descriptive error correlations and nuisance-adjusted correlations."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import sobel
from scipy.stats import rankdata


def pooled(image, size=8):
    height, width = image.shape
    return image.reshape(height // size, size, width // size, size).mean((1, 3)).ravel()


def correlation(x, y):
    if x.std() < 1e-10 or y.std() < 1e-10:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    rows = []
    for file in sorted(args.directory.glob('*/maps.npz')):
        with np.load(file) as archive:
            data = {name: archive[name] for name in archive.files}
            prediction = data['prediction'].clip(0, 1)
            for view in range(3):
                for frame in range(prediction.shape[1]):
                    gray = prediction[view, frame].mean(-1)
                    edge = np.hypot(sobel(gray, axis=0), sobel(gray, axis=1))
                    activity = data['predicted_motion'][view, frame]
                    target = rankdata(pooled(data['pixel_error'][view, frame]))
                    covariates = np.stack([np.ones_like(target),
                                           rankdata(pooled(activity)), rankdata(pooled(edge))], axis=1)
                    # Partial Spearman: linearly remove ranked motion and image
                    # edges from both ranked variables within one view/frame.
                    residual_target = target - covariates @ np.linalg.lstsq(covariates, target, rcond=None)[0]
                    for name in data:
                        if name in ('pixel_error', 'predicted_motion') or data[name].shape != data['pixel_error'].shape:
                            continue
                        score = rankdata(pooled(data[name][view, frame]))
                        residual_score = score - covariates @ np.linalg.lstsq(covariates, score, rcond=None)[0]
                        rows.append({'clip': file.parent.name, 'view': view, 'frame': frame,
                                     'method': name, 'rho_error': correlation(score, target),
                                     'partial_rho_motion_edges': correlation(residual_score, residual_target)})
    if not rows:
        parser.error('No completed maps.npz files found')
    summary = {}
    for method in sorted({row['method'] for row in rows}):
        group = [row for row in rows if row['method'] == method]
        summary[method] = {'n_view_frames': len(group)}
        for metric in ('rho_error', 'partial_rho_motion_edges'):
            values = [row[metric] for row in group if row[metric] is not None]
            summary[method]['median_' + metric] = float(np.median(values)) if values else None
        summary[method]['by_view'] = {}
        for view in range(3):
            values = [row['partial_rho_motion_edges'] for row in group
                      if row['view'] == view and row['partial_rho_motion_edges'] is not None]
            summary[method]['by_view'][str(view)] = float(np.median(values)) if values else None
    output = {'warning': 'Descriptive pilot correlations with RGB error, not hallucination labels. '
                         'Frames are correlated; no significance or calibration claim.',
              'n_clips': len({row['clip'] for row in rows}), 'summary': summary, 'rows': rows}
    (args.directory / 'summary.json').write_text(json.dumps(output, indent=2, allow_nan=False))
    print(json.dumps({key: value for key, value in output.items() if key != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
