import argparse
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile
from io import TextIOWrapper
import json
import logging
import numpy as np

from benchmark.utils import load_poses, subsample_poses, load_K, precision_recall
from benchmark.metrics import MetricManager, Inputs
import benchmark.config as config
from config.default import cfg

def plot_perfect_curve(P):
    total_bins = 1000
    prec_values = []
    ratio_values = []
    for i in range(total_bins):
        ratio_tmp = i/total_bins
        value = min(1, P / ratio_tmp)
        prec_values.append(value)
        ratio_values.append(ratio_tmp)
    return prec_values, ratio_values

def compute_scene_metrics(dataset_path: Path, submission_zip: ZipFile, scene: str):
    metric_manager = MetricManager()

    # load intrinsics and poses
    try:
        K, W, H = load_K(dataset_path / scene / 'intrinsics.txt')
        with (dataset_path / scene / 'poses.txt').open('r', encoding='utf-8') as gt_poses_file:
            gt_poses = load_poses(gt_poses_file, load_confidence=False)
    except FileNotFoundError as e:
        logging.error(f'Could not find ground-truth dataset files: {e}')
        raise
    else:
        logging.info(
            f'Loaded ground-truth intrinsics and poses for scene {scene}')

    # try to load estimated poses from submission
    try:
        with submission_zip.open(f'pose_{scene}.txt') as estimated_poses_file:
            estimated_poses_file_wrapper = TextIOWrapper(
                estimated_poses_file, encoding='utf-8')
            estimated_poses = load_poses(
                estimated_poses_file_wrapper, load_confidence=True)
    except KeyError as e:
        logging.warning(
            f'Submission does not have estimates for scene {scene}.')
        return dict(), len(gt_poses)
    except UnicodeDecodeError as e:
        logging.error('Unsupported file encoding: please use UTF-8')
        raise
    else:
        logging.info(f'Loaded estimated poses for scene {scene}')

    # The val/test set is subsampled by a factor of 5
    gt_poses = subsample_poses(gt_poses, subsample=5)

    # failures encode how many frames did not have an estimate
    # e.g. user/method did not provide an estimate for that frame
    # it's different from when an estimate is provided with low confidence!
    failures = 0

    # Results encoded as dict
    # key: metric name; value: list of values (one per frame).
    # e.g. results['t_err'] = [1.2, 0.3, 0.5, ...]
    results = defaultdict(list)

    # compute metrics per frame
    for frame_num, (q_gt, t_gt, _) in gt_poses.items():
        if frame_num not in estimated_poses:
            failures += 1
            continue

        q_est, t_est, confidence = estimated_poses[frame_num]
        inputs = Inputs(q_gt=q_gt, t_gt=t_gt, q_est=q_est, t_est=t_est,
                        confidence=confidence, K=K[frame_num], W=W, H=H)
        metric_manager(inputs, results)

    return results, failures


def aggregate_results(all_results, all_failures):
    # aggregate metrics
    median_metrics = defaultdict(list)
    all_metrics = defaultdict(list)
    for scene_results in all_results.values():
        for metric, values in scene_results.items():
            median_metrics[metric].append(np.median(values))
            all_metrics[metric].extend(values)
    all_metrics = {k: np.array(v) for k, v in all_metrics.items()}
    assert all([v.ndim == 1 for v in all_metrics.values()]
               ), 'invalid metrics shape'

    # compute avg median metrics
    avg_median_metrics = {metric: np.mean(
        values) for metric, values in median_metrics.items()}

    # compute precision/AUC for pose error and reprojection errors
    accepted_poses = (all_metrics['trans_err'] < config.t_threshold) * \
        (all_metrics['rot_err'] < config.R_threshold)
    accepted_vcre = all_metrics['reproj_err'] < config.vcre_threshold
    total_samples = len(next(iter(all_metrics.values()))) + all_failures

    prec_pose = np.sum(accepted_poses) / total_samples
    prec_vcre = np.sum(accepted_vcre) / total_samples

    # compute AUC for pose and VCRE
    pose_prec_values, pose_recall_values, auc_pose = precision_recall(
        inliers=all_metrics['confidence'], tp=accepted_poses, failures=all_failures)
    vcre_prec_values, vcre_recall_values, auc_vcre = precision_recall(
        inliers=all_metrics['confidence'], tp=accepted_vcre, failures=all_failures)

    curves_data = {}
    curves_data['vcre_prec_values'], curves_data['vcre_recall_values'] = vcre_prec_values, vcre_recall_values
    curves_data['pose_prec_values'], curves_data['pose_recall_values'] = pose_prec_values, pose_recall_values

    # output metrics
    output_metrics = dict()
    output_metrics['Average Median Translation Error'] = avg_median_metrics['trans_err']
    output_metrics['Average Median Rotation Error'] = avg_median_metrics['rot_err']
    output_metrics['Average Median Reprojection Error'] = avg_median_metrics['reproj_err']
    output_metrics[f'Precision @ Pose Error < ({config.t_threshold*100}cm, {config.R_threshold}deg)'] = prec_pose
    output_metrics[f'AUC @ Pose Error < ({config.t_threshold*100}cm, {config.R_threshold}deg)'] = auc_pose
    output_metrics[f'Precision @ VCRE < {config.vcre_threshold}px'] = prec_vcre
    output_metrics[f'AUC @ VCRE < {config.vcre_threshold}px'] = auc_vcre
    output_metrics[f'Estimates for % of frames'] = len(all_metrics['trans_err']) / total_samples
    return output_metrics, curves_data


def count_unexpected_scenes(scenes: tuple, submission_zip: ZipFile):
    submission_scenes = [fname[5:-4]
                         for fname in submission_zip.namelist() if fname.startswith("pose_")]
    return len(set(submission_scenes) - set(scenes))

def main(args):
    dataset_path = args.dataset_path / args.split
    scenes = tuple(f.name for f in dataset_path.iterdir() if f.is_dir())

    try:
        submission_zip = ZipFile(args.submission_path, 'r')
    except FileNotFoundError as e:
        logging.error(f'Could not find ZIP file in path {args.submission_path}')
        return

    all_results = dict()
    all_failures = 0
    for scene in scenes:
        metrics, failures = compute_scene_metrics(
            dataset_path, submission_zip, scene)
        all_results[scene] = metrics
        all_failures += failures

    if all_failures > 0:
        logging.warning(
            f'Submission is missing pose estimates for {all_failures} frames')

    unexpected_scene_count = count_unexpected_scenes(scenes, submission_zip)
    if unexpected_scene_count > 0:
        logging.warning(
            f'Submission contains estimates for {unexpected_scene_count} scenes outside the {args.split} set')

    if all((len(metrics) == 0 for metrics in all_results.values())):
        logging.error(
            f'Submission does not have any valid pose estimates')
        return

    output_metrics, curves_data = aggregate_results(all_results, all_failures)
    output_json = json.dumps(output_metrics, indent=2)
    print(output_json)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'eval', description='Evaluate submissions for the MapFree dataset benchmark')
    parser.add_argument('--submission_path', type=Path, default='',
                        help='Path to the submission ZIP file')
    parser.add_argument('--split', choices=('val', 'test'), default='test',
                        help='Dataset split to use for evaluation. Default: test')
    parser.add_argument('--log', choices=('warning', 'info', 'error'),
                        default='warning', help='Logging level. Default: warning')
    parser.add_argument('--dataset_path', type=Path, default=None,
                        help='Path to the dataset folder')

    args = parser.parse_args()

    if args.dataset_path is None:
        cfg.merge_from_file('config/datasets/mapfree.yaml')
        args.dataset_path = Path(cfg.DATASET.DATA_ROOT)

    logging.basicConfig(level=args.log.upper())
    try:
        main(args)
    except Exception:
        logging.error("Unexpected behaviour. Exiting.")

