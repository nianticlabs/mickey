from dataclasses import dataclass
from typing import Callable

import numpy as np

from benchmark.reprojection import reprojection_error
from benchmark.utils import VARIANTS_ANGLE_SIN, quat_angle_error


@dataclass
class Inputs:
    q_gt: np.array
    t_gt: np.array
    q_est: np.array
    t_est: np.array
    confidence: float
    K: np.array
    W: int
    H: int

    def __post_init__(self):
        assert self.q_gt.shape == (4,), 'invalid gt quaternion shape'
        assert self.t_gt.shape == (3,), 'invalid gt translation shape'
        assert self.q_est.shape == (4,), 'invalid estimated quaternion shape'
        assert self.t_est.shape == (3,), 'invalid estimated translation shape'
        assert self.confidence >= 0, 'confidence must be non negative'
        assert self.K.shape == (3, 3), 'invalid K shape'
        assert self.W > 0, 'invalid image width'
        assert self.H > 0, 'invalid image height'


class MyDict(dict):
    def register(self, fn) -> Callable:
        """Registers a function within dict(fn_name -> fn_ref).
        This is used to evaluate all registered metrics in MetricManager.__call__()"""
        self[fn.__name__] = fn
        return fn


class MetricManager:
    _metrics = MyDict()

    def __call__(self, inputs: Inputs, results: dict) -> None:
        for metric, metric_fn in self._metrics.items():
            results[metric].append(metric_fn(inputs))

    @staticmethod
    @_metrics.register
    def trans_err(inputs: Inputs) -> np.float64:
        return np.linalg.norm(inputs.t_est - inputs.t_gt)

    @staticmethod
    @_metrics.register
    def rot_err(inputs: Inputs, variant: str = VARIANTS_ANGLE_SIN) -> np.float64:
        return quat_angle_error(label=inputs.q_est, pred=inputs.q_gt, variant=variant)[0, 0]

    @staticmethod
    @_metrics.register
    def reproj_err(inputs: Inputs) -> float:
        return reprojection_error(
            q_est=inputs.q_est, t_est=inputs.t_est, q_gt=inputs.q_gt, t_gt=inputs.t_gt, K=inputs.K,
            W=inputs.W, H=inputs.H)

    @staticmethod
    @_metrics.register
    def confidence(inputs: Inputs) -> float:
        return inputs.confidence
