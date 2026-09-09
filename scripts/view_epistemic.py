"""Build a standalone frame-and-method viewer from pilot maps.npz files."""

import argparse
import base64
import io
import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image


def png(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    clips = []
    for path in sorted(args.directory.glob('*/maps.npz')):
        with np.load(path) as arrays:
            prediction = (arrays['prediction'].clip(0, 1) * 255).round().astype('uint8')
            truth = (arrays['ground_truth'].clip(0, 1) * 255).round().astype('uint8')
            maps = {}
            for name in arrays.files:
                if name in ('prediction', 'ground_truth', 'hold_prediction'):
                    continue
                value = arrays[name]
                if value.shape != prediction.shape[:-1]:
                    continue
                scale = max(float(np.quantile(value, .99)), 1e-12)
                colored = matplotlib.colormaps['magma']((value / scale).clip(0, 1), bytes=True)[..., :3]
                label = {'pixel_error': 'Pixel error (evaluation only)',
                         'predicted_motion': 'Predicted motion (control)'}.get(name, name.replace('_', ' '))
                maps[label] = {
                    'scale': scale,
                    'images': [[png(frame) for frame in view] for view in colored]}
            clips.append({'name': path.parent.name,
                          'prediction': [[png(frame) for frame in view] for view in prediction],
                          'truth': [[png(frame) for frame in view] for view in truth],
                          'maps': maps})
    if not clips:
        parser.error('No */maps.npz files found')
    document = '''<!doctype html><html lang="en"><meta charset="utf-8">
<title>140k uncertainty pilot</title><style>
body{font:16px system-ui;background:#15161a;color:#eee;margin:24px;max-width:1200px}
label{display:inline-block;margin:12px 20px 12px 0}select,input{vertical-align:middle}
.views{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
canvas{width:100%;image-rendering:auto}p{line-height:1.5;color:#ccc}small{color:#ccc}
</style><h1>140k uncertainty pilot</h1>
<p>These are diagnostic maps, not calibrated hallucination probabilities.
Color scales are fixed within each clip and method at the 99th percentile.
Pixel error uses the logged future and is available only for evaluation.</p>
<label>Clip <select id="clip"></select></label>
<label>Map <select id="method"></select></label><br>
<label>Future frame <input id="frame" type="range" min="0" value="0"><span id="number"></span></label>
<label>Overlay opacity <input id="opacity" type="range" min="0" max="1" step="0.05" value="0.65"></label>
<p id="scale"></p><div class="views" id="views"></div>
<p>The first row shows generated frames with the selected map; the second shows
the recorded future. Brighter colors indicate larger scores. Parameter and
action-effect maps propagate weight sensitivity through the decoder. The latent
VAE residual is displayed at its coarse latent resolution.</p>
<script>const clips=__DATA__;
const cache=new Map();
function img(url){if(!cache.has(url)){const im=new Image();im.src=url;cache.set(url,im)}return cache.get(url)}
const names=['Top camera','Left wrist','Right wrist'];
const views=document.getElementById('views');
names.forEach((name,i)=>{const box=document.createElement('div');box.innerHTML=
`<h2>${name}</h2><small>Prediction and diagnostic map</small><canvas id="pred${i}"></canvas>
<small>Recorded future</small><canvas id="gt${i}"></canvas>`;views.append(box)});
const select=document.getElementById('clip'),method=document.getElementById('method'),
frame=document.getElementById('frame'),opacity=document.getElementById('opacity');
clips.forEach((c,i)=>select.add(new Option(c.name,i)));
let revision=0;
async function draw(){const rev=++revision,c=clips[Number(select.value)],f=Number(frame.value),m=c.maps[method.value];
document.getElementById('number').textContent=' '+(f+1)+' / '+c.prediction[0].length;
document.getElementById('scale').textContent='Scale: 0 to '+m.scale.toPrecision(4)+' (clipped above this value).';
for(let v=0;v<3;v++){const p=img(c.prediction[v][f]),g=img(c.truth[v][f]),u=img(m.images[v][f]);
await Promise.all([p.decode(),g.decode(),u.decode()]);if(rev!==revision)return;
const cp=document.getElementById('pred'+v),cg=document.getElementById('gt'+v);
cp.width=cg.width=p.width;cp.height=cg.height=p.height;
let ctx=cp.getContext('2d');ctx.drawImage(p,0,0);ctx.globalAlpha=Number(opacity.value);ctx.drawImage(u,0,0);
cg.getContext('2d').drawImage(g,0,0);}}
function choose(){const c=clips[Number(select.value)],old=method.value;method.replaceChildren();
Object.keys(c.maps).forEach(key=>method.add(new Option(key,key)));
if(c.maps[old])method.value=old;else if(c.maps['Parameter variance'])method.value='Parameter variance';
frame.max=c.prediction[0].length-1;frame.value=Math.min(Number(frame.value),Number(frame.max));draw()}
select.onchange=choose;method.onchange=draw;frame.oninput=draw;opacity.oninput=draw;choose();
</script></html>'''
    metadata_path = args.directory / 'metadata.json'
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        fit = metadata.get('fit', {})
        eigenvalues = fit.get('covariance_eigenvalues', [])
        checks = fit.get('fit_fd_relative_error', [])
        if eigenvalues and checks:
            diagnostic = (f'<p><strong>Fit diagnostics:</strong> posterior coefficient variance '
                          f'ranges from {min(eigenvalues):.3f} to {max(eigenvalues):.3f} '
                          f'(prior = 1). Halving the training finite-difference step changes '
                          f'the Jacobian by {checks[0]:.1%}. These are experimental sensitivity '
                          f'maps; the small fit does not establish reliable epistemic uncertainty.</p>')
            document = document.replace('<label>Clip ', diagnostic + '<label>Clip ', 1)
    destination = args.directory / 'viewer.html'
    destination.write_text(document.replace('__DATA__', json.dumps(clips).replace('</', '<\\/')))
    print(destination)


if __name__ == '__main__':
    main()
