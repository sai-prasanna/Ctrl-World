import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple
from tqdm import tqdm
import torch
import random
import imageio
from decord import VideoReader, cpu
from accelerate.logging import get_logger
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
from typing_extensions import override
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
# from finetune.constants import LOG_LEVEL, LOG_NAME
import numpy as np
from scipy.spatial.transform import Rotation as R  

def load_and_process_ann_file(data_root, ann_file, sequence_interval=1, start_interval=4,
                              sequence_length=8, keep_states=False):
    samples = []
    try:
        with open(f'{data_root}/{ann_file}', "r") as f:
            ann = json.load(f)
    except:
        print(f'skip {ann_file}')
        return samples

    n_frames = ann['video_length']
    traj_len = int(sequence_length*sequence_interval)
    end_idx = n_frames - int(traj_len*0.5)
    if end_idx < 1:
        end_idx = 1

    for start_frame in range(0,end_idx,start_interval):       
        idx = start_frame
        sample = dict()
        sample['episode_id'] = ann['episode_id']
        sample['frame_ids'] = [idx]
        if keep_states:
            # one numpy array per sample; only needed for --write_stat, and there are
            # millions of samples, so keep it off by default (it OOMs otherwise)
            sample['states'] = np.array(ann['states'])[idx:idx+1]
        samples.append(sample)
    return samples

def init_anns(dataset_root, data_dir):
    final_path = f'{dataset_root}/{data_dir}'
    ann_files = [os.path.join(data_dir, f) for f in os.listdir(final_path) if f.endswith('.json')]
    return ann_files

def init_sequences(data_root, ann_files, sequence_interval, start_interval,sequence_length,
                   keep_states=False):
    samples = []
    with ThreadPoolExecutor(32) as executor:
        future_to_ann_file = {executor.submit(load_and_process_ann_file, data_root, ann_file, sequence_interval, start_interval, sequence_length, keep_states): ann_file for ann_file in ann_files}
        for future in tqdm(as_completed(future_to_ann_file), total=len(ann_files)):
            samples.extend(future.result())
    return samples


if __name__ == "__main__":

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--droid_output_path', type=str, default='dataset_example/droid_subset')
    # dataset_name
    parser.add_argument('--dataset_name', type=str, default='droid_subset')
    # Where the index lands. It defaults into the repo, which is fine for a local run, but
    # a `cluster submit` checkout is thrown away after the job, and train_sample.json runs
    # to hundreds of MB — point this at durable storage there.
    parser.add_argument('--meta_root', type=str, default='dataset_meta_info')
    parser.add_argument('--write_stat', action='store_true',
                        help='recompute stat.json from the annotations (overwrites)')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    ########################### xhand datasets ###########################
    sequence_length = 8
    for data_type in ['val', 'train']:
        samples_all = []
        ann_files_all = []
        data_root = args.droid_output_path
        dataset_name = args.dataset_name

        sequence_interval = 1
        start_interval = 1
        ann_dir = f'annotation/{data_type}'
        ann_files = init_anns(data_root, ann_dir)
        ann_files_all.extend(ann_files)
        samples = init_sequences(data_root, ann_files,sequence_interval, start_interval, sequence_length,
                                 keep_states=args.write_stat)
        print(f'{data_root} {len(samples)} samples')
        samples_all.extend(samples)
        
        # calculate the 1% and 99% of the state, used by Dataset_mix.normalize_bound.
        # Opt-in: for ABC, select_abc_episodes.py already derives stat.json from the
        # dataset's own per-episode q01/q99, which is more accurate than this subsample.
        if args.write_stat and data_type == 'train':
            print("########################### state ###########################")
            state_all = np.concatenate([np.array(s['states']).reshape(1, -1) for s in samples_all], axis=0)
            print(state_all.shape)
            state_01 = np.percentile(state_all, 1, axis=0)
            state_99 = np.percentile(state_all, 99, axis=0)
            print('state_01:', state_01)
            print('state_99:', state_99)
            os.makedirs(f'{args.meta_root}/{dataset_name}', exist_ok=True)
            with open(f'{args.meta_root}/{dataset_name}/stat.json', 'w') as f:
                json.dump({'state_01': state_01.tolist(), 'state_99': state_99.tolist()}, f, indent=2)

        
        # dataset meta info
        for samples in samples_all:
            samples.pop('states', None)
        import random
        random.shuffle(samples_all)
        print('step_num',data_type,len(samples_all))
        print('traj_num',data_type, len(ann_files_all))
        os.makedirs(f'{args.meta_root}/{dataset_name}', exist_ok=True)
        with open(f'{args.meta_root}/{dataset_name}/{data_type}_sample.json', 'w') as f:
            json.dump(samples_all, f)
        
