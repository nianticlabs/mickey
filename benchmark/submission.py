import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from zipfile import ZipFile

import torch
import numpy as np
from tqdm import tqdm

from config.default import cfg
from lib.datasets.datamodules import DataModule
from lib.models.builder import build_model
from lib.utils.data import data_to_model_device
from transforms3d.quaternions import mat2quat

@dataclass
class Pose:
    image_name: str
    q: np.ndarray
    t: np.ndarray
    inliers: float

    def __str__(self) -> str:
        formatter = {'float': lambda v: f'{v:.6f}'}
        max_line_width = 1000
        q_str = np.array2string(self.q, formatter=formatter, max_line_width=max_line_width)[1:-1]
        t_str = np.array2string(self.t, formatter=formatter, max_line_width=max_line_width)[1:-1]
        return f'{self.image_name} {q_str} {t_str} {self.inliers}'


def predict(loader, model):
    results_dict = defaultdict(list)

    for data in tqdm(loader):

        # run inference
        data = data_to_model_device(data, model)
        with torch.no_grad():
            R_batched, t_batched = model(data)

        for i_batch in range(len(data['scene_id'])):
            R = R_batched[i_batch].unsqueeze(0).detach().cpu().numpy()
            t = t_batched[i_batch].reshape(-1).detach().cpu().numpy()
            inliers = data['inliers'][i_batch].item()

            scene = data['scene_id'][i_batch]
            query_img = data['pair_names'][1][i_batch]

            # ignore frames without poses (e.g. not enough feature matches)
            if np.isnan(R).any() or np.isnan(t).any() or np.isinf(t).any():
                continue

            # populate results_dict
            estimated_pose = Pose(image_name=query_img,
                                  q=mat2quat(R).reshape(-1),
                                  t=t.reshape(-1),
                                  inliers=inliers)
            results_dict[scene].append(estimated_pose)

    return results_dict


def save_submission(results_dict: dict, output_path: Path):
    with ZipFile(output_path, 'w') as zip:
        for scene, poses in results_dict.items():
            poses_str = '\n'.join((str(pose) for pose in poses))
            zip.writestr(f'pose_{scene}.txt', poses_str.encode('utf-8'))


def eval(args):
    # Load configs
    cfg.merge_from_file('config/datasets/mapfree.yaml')
    cfg.merge_from_file(args.config)

    # Create dataloader
    if args.split == 'test':
        cfg.TRAINING.BATCH_SIZE = 8
        cfg.TRAINING.NUM_WORKERS = 8
        dataloader = DataModule(cfg, drop_last_val=False).test_dataloader()
    elif args.split == 'val':
        cfg.TRAINING.BATCH_SIZE = 16
        cfg.TRAINING.NUM_WORKERS = 8
        dataloader = DataModule(cfg, drop_last_val=False).val_dataloader()
    else:
        raise NotImplemented(f'Invalid split: {args.split}')

    # Create model
    model = build_model(cfg, args.checkpoint)

    # Get predictions from model
    results_dict = predict(dataloader, model)

    # Save predictions to txt per scene within zip
    args.output_root.mkdir(parents=True, exist_ok=True)
    save_submission(results_dict, args.output_root / 'submission.zip')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='path to config file')
    parser.add_argument('--checkpoint',
                        help='path to model checkpoint (models with learned parameters)', default='')
    parser.add_argument('--output_root', '-o', type=Path, default=Path('results/'))
    parser.add_argument('--split', choices=('val', 'test'), default='test',
                        help='Dataset split to use for evaluation. Choose from test or val. Default: test')
    args = parser.parse_args()
    eval(args)
