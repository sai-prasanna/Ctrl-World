"""Serve a Ctrl-World checkpoint to the benchmark environment over a Unix socket.

Two virtual environments, one node. Ctrl-World pins torch 2.7.1 and diffusers; the
benchmark's metrics, judge and ABC policy want torch 2.11 and a pinned `abc` checkout, and
no resolver satisfies both. Rather than choosing, the checkpoint stays in `venv` behind
this server and everything that scores it runs in `venv_bench` against
`wmbench.ipc.wm_client.RemoteWorldModel`, which satisfies the same `WorldModel` protocol
as the in-process bridge.

Thin on purpose. Every decision about what the model is - the config, the history pattern,
the rollout shape - is `bridges/ctrlworld_wm.py`'s and is made here exactly as
`scripts/wmbench_rollout_ctrlworld.py` makes it, so a rollout through the socket and a
rollout in process differ in nothing but the transport. The flags are that script's.

    # in the model's environment, on the GPU node
    python3 scripts/wmbench_wm_server.py \\
        --ckpt_path outputs/0003_abc_mcap/model/checkpoint-200000.pt \\
        --clips dataset_meta_info/abc_mcap/eval_clips_v1.json \\
        --val_dataset_dir $CTRLWORLD_DATA/abc_mcap \\
        --data_stat_path $CTRLWORLD_META/abc_mcap/stat.json \\
        --socket $TMPDIR/ctrlworld.sock --ready_file $TMPDIR/ctrlworld.ready

    # in venv_bench, once the ready file appears
    wmbench rollout --world-model remote \\
        --world-model-arg socket=$TMPDIR/ctrlworld.sock ...

The ready file is the handshake. Loading a checkpoint takes minutes, and the socket path
exists from `bind`, before `listen` accepts, so a client that waits on the socket alone
races the load.
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bridges.ctrlworld_wm import HISTORY_IDX, CtrlWorldModel  # noqa: E402
from wmbench.ipc.wm_server import serve  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_path', type=str, required=True)
    p.add_argument('--socket', type=str, required=True,
                   help='Unix socket to bind; keep it short, an address holds 107 bytes')
    p.add_argument('--ready_file', type=str, default=None,
                   help='touched once the socket accepts, so the client waits for the '
                        'checkpoint to load instead of racing it')
    p.add_argument('--clips', type=str, default=None,
                   help='clip list; read only for its rollout shape (pred_step, '
                        'interact_num), which the client sees through `describe`')
    p.add_argument('--val_dataset_dir', type=str, default=None,
                   help='the bridge reads its conditioning latents from here itself')
    p.add_argument('--data_stat_path', type=str, default=None)
    p.add_argument('--svd_model_path', type=str, default=None)
    p.add_argument('--clip_model_path', type=str, default=None)
    p.add_argument('--split', type=str, default='val')
    p.add_argument('--pred_step', type=int, default=None,
                   help='frames per round; defaults to the clip list\'s')
    p.add_argument('--interact_num', type=int, default=None)
    p.add_argument('--teacher_forced', action='store_true')
    p.add_argument('--history_idx', type=str, default=None,
                   help='comma-separated history buffer indexes; defaults to the uniform '
                        '"-6,-5,-4,-3,-2,-1". The authors use "0,0,-12,-9,-6,-3".')
    p.add_argument('--device', type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    from config import merge_args, wm_args

    history_idx = ([int(x) for x in args.history_idx.split(',')] if args.history_idx
                   else HISTORY_IDX)
    cfg = wm_args(task_type='abc_replay')
    cfg = merge_args(cfg, args)
    # The rollout shape has to match the clip list the client will drive with: a list
    # drawn for 12 rounds of 5 frames reserves exactly that much ground truth per clip.
    # Taking it from the list rather than from config.py is what keeps a socket rollout
    # and an in-process one comparable.
    if args.clips:
        payload = json.load(open(args.clips))
        cfg.pred_step = args.pred_step or payload['pred_step']
        cfg.interact_num = args.interact_num or payload['interact_num']
        if args.val_dataset_dir is None:
            args.val_dataset_dir = payload['val_dataset_dir']
    else:
        cfg.pred_step = args.pred_step or cfg.pred_step
        cfg.interact_num = args.interact_num or cfg.interact_num

    wm = CtrlWorldModel(cfg, ckpt_path=args.ckpt_path,
                        val_dataset_dir=args.val_dataset_dir,
                        data_stat_path=args.data_stat_path or cfg.data_stat_path,
                        split=args.split, history_idx=history_idx,
                        teacher_forced=args.teacher_forced, device=args.device)
    op = wm.operating_point
    print(f'loaded {args.ckpt_path}: {op.width}x{op.height} at {op.fps} Hz, '
          f'{wm.frames_per_step} frames per step, preprocess {op.preprocess_id}',
          flush=True)
    serve(wm, args.socket, args.ready_file)


if __name__ == '__main__':
    main()
