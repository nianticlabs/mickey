# Code adapted from Map-free benchmark: https://github.com/nianticlabs/map-free-reloc

import torch
import numpy as np
from collections import defaultdict
from lib.benchmarks.reprojection import get_grid_multipleheight
from lib.models.MicKey.modules.utils.training_utils import project_2d

# global variable, avoids creating it again
eye_coords_glob = get_grid_multipleheight()

def pose_error_torch(R, t, Tgt, reduce=None):
    """Compute angular, scale and euclidean error of translation vector (metric). Compute angular rotation error."""

    Rgt = Tgt[:, :3, :3]                  # [B, 3, 3]
    tgt = Tgt[:, :3, 3:].transpose(1, 2)  # [B, 1, 3]

    scale_t = torch.linalg.norm(t, dim=-1)
    scale_tgt = torch.linalg.norm(tgt, dim=-1)

    cosine = (t @ tgt.transpose(1, 2)).squeeze(-1) / (scale_t * scale_tgt + 1e-9)
    cosine = torch.clip(cosine, -1.0, 1.0)    # handle numerical errors
    t_ang_err = torch.rad2deg(torch.acos(cosine))
    t_ang_err = torch.minimum(t_ang_err, 180 - t_ang_err)

    t_scale_err = scale_t / scale_tgt
    t_scale_err_sym = torch.maximum(scale_t / scale_tgt, scale_tgt / scale_t)
    t_euclidean_err = torch.linalg.norm(t - tgt, dim=-1)

    residual = R.transpose(1, 2) @ Rgt
    trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
    cosine = (trace - 1) / 2
    cosine = torch.clip(cosine, -1., 1.)  # handle numerical errors
    R_err = torch.rad2deg(torch.acos(cosine))

    if reduce is None:
        def fn(x): return x
    elif reduce == 'mean':
        fn = torch.mean
    elif reduce == 'median':
        fn = torch.median

    t_ang_err = fn(t_ang_err)
    t_scale_err = fn(t_scale_err)
    t_euclidean_err = fn(t_euclidean_err)
    R_err = fn(R_err)

    errors = {'t_err_ang': t_ang_err,
              't_err_scale': t_scale_err,
              't_err_scale_sym': t_scale_err_sym,
              't_err_euc': t_euclidean_err,
              'R_err': R_err}
    return errors


def vcre_loss(R, t, Tgt, K0, H=720):
    """Compute Virtual Correspondences Reprojection Error in torch (with batches)."""

    B = R.shape[0]
    Rgt = Tgt[:, :3, :3]                  # [B, 3, 3]
    tgt = Tgt[:, :3, 3:].transpose(1, 2)  # [B, 1, 3]

    eye_coords = torch.from_numpy(eye_coords_glob).unsqueeze(0)[:, :, :3].to(R.device).float()
    eye_coords = torch.tile(eye_coords, [B, 1, 1])

    # obtain ground-truth position of projected points
    uv_gt = project_2d(eye_coords, K0)

    # Avoid breaking gradients due to inplace operation
    eye_coord_tmp = (R @ eye_coords.transpose(2, 1) + t.transpose(2, 1))
    eyes_residual = (Rgt.transpose(2, 1) @ eye_coord_tmp -1 * Rgt.transpose(2, 1) @ tgt.transpose(2, 1)).transpose(2, 1)

    uv_pred = project_2d(eyes_residual, K0)

    uv_gt = torch.clip(uv_gt, 0, H)
    uv_pred = torch.clip(uv_pred, 0, H)

    repr_err = ((((uv_gt - uv_pred) ** 2.).sum(-1) + 1e-6) ** 0.5).mean(-1).view(B, 1)

    return repr_err


def vcre_torch(R, t, Tgt, K0, reduce=None, H=720, W=540):
    """Compute Virtual Correspondences Reprojection Error in torch (with batches)."""

    B = R.shape[0]
    Rgt = Tgt[:, :3, :3]                  # [B, 3, 3]
    tgt = Tgt[:, :3, 3:].transpose(1, 2)  # [B, 1, 3]

    eye_coords = torch.from_numpy(eye_coords_glob).unsqueeze(0).to(R.device).float()
    eye_coords = torch.tile(eye_coords, [B, 1, 1])

    # obtain ground-truth position of projected points
    uv_gt = project_2d(eye_coords[:, :, :3], K0)

    # residual transformation
    cam2w_est = torch.tile(torch.eye(4).view(1, 4, 4), [B, 1, 1]).to(R.device).float()
    cam2w_est[:, :3, :3] = R
    cam2w_est[:, :3, -1] = t[:, 0]

    cam2w_gt = torch.tile(torch.eye(4).view(1, 4, 4), [B, 1, 1]).to(R.device).float()
    cam2w_gt[:, :3, :3] = Rgt
    cam2w_gt[:, :3, -1] = tgt[:, 0]

    # residual reprojection
    eyes_residual = (torch.linalg.inv(cam2w_gt) @ cam2w_est @ eye_coords.transpose(2, 1)).transpose(2, 1)
    uv_pred = project_2d(eyes_residual[:, :, :3], K0)

    uv_gt[:, :, 0], uv_pred[:, :, 0] = torch.clip(uv_gt[:, :, 0], 0, W), torch.clip(uv_pred[:, :, 0], 0, W)
    uv_gt[:, :, 1], uv_pred[:, :, 1] = torch.clip(uv_gt[:, :, 1], 0, H), torch.clip(uv_pred[:, :, 1], 0, H)

    repr_err = ((((uv_gt - uv_pred) ** 2.).sum(-1) + 1e-6) ** 0.5).mean(-1).view(B, 1)

    if reduce is None:
        def fn(x): return x
    elif reduce == 'mean':
        fn = torch.mean
    elif reduce == 'median':
        fn = torch.median

    repr_err = fn(repr_err)

    errors = {'repr_err': repr_err}

    return errors



def error_auc(errors, thresholds):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    errors = np.nan_to_num(errors, nan=float('inf'))   # convert nans to inf
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}


def ecdf(x):
    """Get Empirical Cumulative Distribution Function (ECDF) given samples x [N,]"""
    cd = np.linspace(0, 1, x.shape[0])
    v = np.sort(x)
    return v, cd


def print_auc_table(agg_metrics):
    pose_error = np.maximum(agg_metrics['R_err'], agg_metrics['t_err_ang'])
    auc_pose = error_auc(pose_error, (5, 10, 20))
    print('Pose error AUC @ 5/10/20deg: {0:.3f}/{1:.3f}/{2:.3f}'.format(*auc_pose.values()))

    auc_rotation = error_auc(agg_metrics['R_err'], (5, 10, 20))
    print('Rotation error AUC @ 5/10/20deg: {0:.3f}/{1:.3f}/{2:.3f}'.format(*auc_rotation.values()))

    auc_translation_ang = error_auc(agg_metrics['t_err_ang'], (5, 10, 20))
    print(
        'Translation angular error AUC @ 5/10/20deg: {0:.3f}/{1:.3f}/{2:.3f}'.format(*auc_translation_ang.values()))

    auc_translation_euc = error_auc(agg_metrics['t_err_euc'], (0.1, 0.5, 1))
    print(
        'Translation Euclidean error AUC @ 0.1/0.5/1m: {0:.3f}/{1:.3f}/{2:.3f}'.format(*auc_translation_euc.values()))


def precision(agg_metrics, rot_threshold, trans_threshold):
    '''Provides ratio of samples with rotation error < rot_threshold AND translation error < trans_threshold'''
    mask_rot = agg_metrics['R_err'] <= rot_threshold
    mask_trans = agg_metrics['t_err_euc'] <= trans_threshold
    recall = (mask_rot * mask_trans).mean()
    return recall


def A_metrics(t_scale_err_sym):
    """Returns A1/A2/A3 metrics of translation vector norm given the "symmetric" scale error
    where
    t_scale_err_sym = torch.maximum((t_norm_gt / t_norm_pred), (t_norm_pred / t_norm_gt))
    """

    if not torch.is_tensor(t_scale_err_sym):
        t_scale_err_sym = torch.from_numpy(t_scale_err_sym)

    thresh = t_scale_err_sym
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()
    return a1, a2, a3


class MetricsAccumulator:
    """Accumulates metrics and aggregates them when requested"""

    def __init__(self):
        self.data = defaultdict(list)

    def accumulate(self, data):
        for key, value in data.items():
            self.data[key].append(value)

    def aggregate(self):
        res = dict()
        for key in self.data.keys():
            res[key] = torch.cat(self.data[key]).view(-1).cpu().numpy()
        return res
