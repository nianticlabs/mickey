import numpy as np
import pytest
from transforms3d.euler import euler2quat
from transforms3d.quaternions import axangle2quat, qmult, quat2mat, rotate_vector

from benchmark.metrics import Inputs, MetricManager
from benchmark.reprojection import project
from benchmark.utils import VARIANTS_ANGLE_COS, VARIANTS_ANGLE_SIN


def createInput(q_gt=None, t_gt=None, q_est=None, t_est=None, confidence=None, K=None, W=None, H=None):
    q_gt = np.zeros(4) if q_gt is None else q_gt
    t_gt = np.zeros(3) if t_gt is None else t_gt
    q_est = np.zeros(4) if q_est is None else q_est
    t_est = np.zeros(3) if t_est is None else t_est
    confidence = 0. if confidence is None else confidence
    K = np.eye(3) if K is None else K
    H = 1 if H is None else H
    W = 1 if W is None else W
    return Inputs(q_gt=q_gt, t_gt=t_gt, q_est=q_est, t_est=t_est, confidence=confidence, K=K, W=W, H=H)


def randomQuat():
    angles = np.random.uniform(0, 2*np.pi, 3)
    q = euler2quat(*angles)
    return q


class TestMetrics:
    @pytest.mark.parametrize('run_number', range(50))
    def test_t_err_tinvariance(self, run_number: int) -> None:
        """Computes the translation error given an initial translation and displacement of this
        translation. The translation error must be equal to the norm of the displacement."""
        mean, var = 5, 10
        t0 = np.random.normal(mean, var, (3,))
        displacement = np.random.normal(mean, var, (3,))

        i = createInput(t_gt=t0, t_est=t0+displacement)
        trans_err = MetricManager.trans_err(i)
        assert np.isclose(trans_err, np.linalg.norm(displacement))

    @pytest.mark.parametrize('run_number', range(50))
    def test_trans_err_rinvariance(self, run_number: int) -> None:
        """Computes the translation error given estimated and gt vectors.
        The translation error must be the same for a rotated version of those vectors
        (same random rotation)"""
        mean, var = 5, 10
        t0 = np.random.normal(mean, var, (3,))
        t1 = np.random.normal(mean, var, (3,))
        q = randomQuat()

        i = createInput(t_gt=t0, t_est=t1)
        trans_err = MetricManager.trans_err(i)

        ir = createInput(t_gt=rotate_vector(t0, q), t_est=rotate_vector(t1, q))
        trans_err_r = MetricManager.trans_err(ir)

        assert np.isclose(trans_err, trans_err_r)

    @pytest.mark.parametrize('run_number', range(50))
    @pytest.mark.parametrize('dtype', (np.float64, np.float32))
    def test_rot_err_raxis(self, run_number: int, dtype: type) -> None:
        """Test rotation error for rotations around a random axis.

        Note: We create GT as high precision, and only downcast when calling rot_err.
        """
        q = randomQuat().astype(np.float64)

        axis = np.random.uniform(low=-1, high=1, size=3).astype(np.float64)
        angle = np.float64(np.random.uniform(low=-np.pi, high=np.pi))
        qres = axangle2quat(vector=axis, theta=angle, is_normalized=False).astype(np.float64)

        i = createInput(q_gt=q.astype(dtype), q_est=qmult(q, qres).astype(dtype))
        rot_err = MetricManager.rot_err(i)
        assert isinstance(rot_err, np.float64)
        rot_err_expected = np.abs(np.degrees(angle))
        # if we add up errors, we want them to be positive
        assert 0. <= rot_err
        rtol = 1.e-5  # numpy default
        atol = 1.e-8  # numpy default
        if isinstance(dtype, np.float32):
            atol = 1.e-7  # 1/50 test might fail at 1.e-8
        assert np.isclose(rot_err, rot_err_expected, rtol=rtol, atol=atol)

    @pytest.mark.parametrize('run_number', range(50))
    def test_r_err_mat(self, run_number: int) -> None:
        q0 = randomQuat()
        q1 = randomQuat()

        i = createInput(q_gt=q0, q_est=q1)
        rot_err = MetricManager.rot_err(i)

        R0 = quat2mat(q0)
        R1 = quat2mat(q1)
        Rres = R1 @ R0.T
        theta = (np.trace(Rres) - 1)/2
        theta = np.clip(theta, -1, 1)
        angle = np.degrees(np.arccos(theta))

        assert np.isclose(angle, rot_err)

    def test_reproj_error_identity(self):
        """Test that reprojection error is zero if poses match"""
        q = randomQuat()
        t = np.random.normal(0, 10, (3,))
        i = createInput(q_gt=q, t_gt=t, q_est=q, t_est=t)

        reproj_err = MetricManager.reproj_err(i)
        assert np.isclose(reproj_err, 0)

    @pytest.mark.parametrize('run_number', range(10))
    @pytest.mark.parametrize('variant', (VARIANTS_ANGLE_SIN,))
    @pytest.mark.parametrize('dtype', (np.float64,))
    def test_r_err_small(self, run_number: int, variant: str, dtype: type) -> None:
        """Test rotation error for small angle differences.

        Note: We create GT as high precision, and only downcast when calling rot_err.
        """
        scales_failed = []
        for scale in np.logspace(start=-1, stop=-9, num=9, base=10, dtype=dtype):
            q = randomQuat().astype(np.float64)
            angle = np.float64(np.random.uniform(low=-np.pi, high=np.pi)) * scale
            assert isinstance(angle, np.float64)
            axis = np.random.uniform(low=-1., high=1., size=3).astype(np.float64)
            assert axis.dtype == np.float64
            qres = axangle2quat(vector=axis, theta=angle, is_normalized=False).astype(np.float64)
            assert qres.dtype == np.float64

            i = createInput(q_gt=q.astype(dtype), q_est=qmult(q, qres).astype(dtype))

            # We expect the error to always be np.float64 for highest acc.
            rot_err = MetricManager.rot_err(i, variant=variant)
            assert isinstance(rot_err, np.float64)
            rot_err_expected = np.abs(np.degrees(angle))
            assert isinstance(rot_err_expected, type(rot_err))

            # if we add up errors, we want them to be positive
            assert 0. <= rot_err

            # check accuracy for one magnitude higher tolerance than the angle
            tol = 0.1 * scale
            # need to be more permissive for lower precision
            if dtype == np.float32:
                tol = 1.e3 * scale

            # cast to dtype for checking
            rot_err = rot_err.astype(dtype)
            rot_err_expected = rot_err_expected.astype(dtype)

            if variant == VARIANTS_ANGLE_SIN:
                assert np.isclose(rot_err, rot_err_expected, rtol=tol, atol=tol)
            elif variant == VARIANTS_ANGLE_COS:
                if not np.isclose(rot_err, rot_err_expected, rtol=tol, atol=tol):
                    print(f"[variant '{variant}'] raises an error for\n"
                          f"\trot_err: {rot_err}"
                          f"\trot_err_expected: {rot_err_expected}"
                          f"\trtol: {tol}"
                          f"\tatol: {tol}")
                    scales_failed.append(scale)
        if len(scales_failed):
            pytest.fail(f"Variant {variant} failed at scales {scales_failed}")


def test_projection() -> None:
    xyz = np.array(((10, 20, 30), (10, 30, 50), (-20, -15, 5),
                   (-20, -50, 10)), dtype=np.float32)
    K = np.eye(3)

    uv = np.array(((1/3, 2/3), (1/5, 3/5), (-4, -3),
                  (-2, -5)), dtype=np.float32)
    assert np.allclose(uv, project(xyz, K))

    uv = np.array(((1/3, 2/3), (1/5, 3/5), (0, 0), (0, 0)), dtype=np.float32)
    assert np.allclose(uv, project(xyz, K, img_size=(5, 5)))
