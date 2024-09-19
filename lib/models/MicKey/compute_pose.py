import pytorch_lightning as pl

from lib.models.MicKey.modules.compute_correspondences import ComputeCorrespondences
from lib.models.MicKey.modules.utils.probabilisticProcrustes import e2eProbabilisticProcrustesSolver

class MickeyRelativePose(pl.LightningModule):
    # Compute the metric relative pose between two input images, with given intrinsics (for the pose solver).

    def __init__(self, cfg):
        super().__init__()

        # Define MicKey architecture and matching module:
        self.compute_matches = ComputeCorrespondences(cfg)

        # Metric solver
        self.e2e_Procrustes = e2eProbabilisticProcrustesSolver(cfg)

        self.is_eval_model(True)

    def forward(self, data):

        self.compute_matches(data)
        data['final_scores'] = data['scores'] * data['kp_scores']

        # Returns inliers list:
        # R, t, inliers, inliers_list = self.e2e_Procrustes.estimate_pose_vectorized(data, return_inliers=True)

        # If the inlier list is not needed:
        R, t, inliers = self.e2e_Procrustes.estimate_pose_vectorized(data, return_inliers=False)

        data['R'] = R
        data['t'] = t
        data['inliers'] = inliers
        # data['inliers_list'] = inliers_list

        return R, t

    def on_load_checkpoint(self, checkpoint):
        # This function avoids loading DINOv2 which are not sotred in Mickey's checkpoint.
        # This saves memory during training, since DINOv2 is frozen and not updated there is no need to store
        # the weights in every checkpoint.

        # Recover DINOv2 features from pretrained weights.
        for param_tensor in self.compute_matches.state_dict():
            if 'dinov2'in param_tensor:
                checkpoint['state_dict']['compute_matches.'+param_tensor] = \
                    self.compute_matches.state_dict()[param_tensor]

    def is_eval_model(self, is_eval):
        if is_eval:
            self.compute_matches.extractor.depth_head.eval()
            self.compute_matches.extractor.det_offset.eval()
            self.compute_matches.extractor.dsc_head.eval()
            self.compute_matches.extractor.det_head.eval()
        else:
            self.compute_matches.extractor.depth_head.train()
            self.compute_matches.extractor.det_offset.train()
            self.compute_matches.extractor.dsc_head.train()
            self.compute_matches.extractor.det_head.train()
