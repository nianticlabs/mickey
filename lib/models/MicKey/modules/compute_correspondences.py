import torch
import torch.nn as nn
from lib.models.MicKey.modules.mickey_extractor import MicKey_Extractor
from lib.models.MicKey.modules.utils.feature_matcher import featureMatcher

class ComputeCorrespondences(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Feature extractor
        self.extractor = MicKey_Extractor(cfg['MICKEY'])

        self.dsc_dim = cfg['MICKEY']['DSC_HEAD']['LAST_DIM']

        # Feature matcher
        self.matcher = featureMatcher(cfg['FEATURE_MATCHER'])

        self.down_factor = cfg['MICKEY']['DINOV2']['DOWN_FACTOR']

    def get_abs_kpts_coordinates(self, kpts):

        B, C, H, W = kpts.shape

        # Compute offset for every kp grid
        x_abs_pos = torch.arange(W).view(1, 1, W).tile([B, H, 1]).to(kpts.device)
        y_abs_pos = torch.arange(H).view(1, H, 1).tile([B, 1, W]).to(kpts.device)
        abs_pos = torch.concat([x_abs_pos.unsqueeze(1), y_abs_pos.unsqueeze(1)], dim=1)

        kpts_abs_pos = (kpts + abs_pos) * self.down_factor

        return kpts_abs_pos

    def prepare_kpts_dsc(self, kpt, depth, scr, dsc):

        B, _, H, W = kpt.shape
        num_kpts = (H * W)

        kpt = kpt.view(B, 2, num_kpts)
        depth = depth.view(B, 1, num_kpts)
        scr = scr.view(B, 1, num_kpts)
        dsc = dsc.view(B, self.dsc_dim, num_kpts)

        return kpt, depth, scr, dsc

    # Independent method to only combine matching and keypoint scores during training
    def kp_matrix_scores(self, sc0, sc1):

        # matrix with "probability" of sampling a correspondence based on keypoint scores only
        scores = torch.matmul(sc0.transpose(2, 1).contiguous(), sc1)
        return scores

    def forward(self, data):

        # Compute detection and descriptor maps
        im0 = data['image0']
        im1 = data['image1']

        # Extract independently features from im0 and im1
        kps0, depth0, scr0, dsc0 = self.extractor(im0)
        kps1, depth1, scr1, dsc1 = self.extractor(im1)

        kps0 = self.get_abs_kpts_coordinates(kps0)
        kps1 = self.get_abs_kpts_coordinates(kps1)

        # Log shape for logging purposes
        _, _, H_kp0, W_kp0 = kps0.shape
        _, _, H_kp1, W_kp1 = kps1.shape
        data['kps0_shape'] = [H_kp0, W_kp0]
        data['kps1_shape'] = [H_kp1, W_kp1]
        data['depth0_map'] = depth0
        data['depth1_map'] = depth1
        data['down_factor'] = self.down_factor

        # Reshape kpts and descriptors to [B, num_kpts, dim]
        kps0, depth0, scr0, dsc0 = self.prepare_kpts_dsc(kps0, depth0, scr0, dsc0)
        kps1, depth1, scr1, dsc1 = self.prepare_kpts_dsc(kps1, depth1, scr1, dsc1)

        # get correspondences
        scores = self.matcher(kps0, dsc0, kps1, dsc1)

        data['kps0'] = kps0
        data['depth_kp0'] = depth0
        data['scr0'] = scr0
        data['kps1'] = kps1
        data['depth_kp1'] = depth1
        data['scr1'] = scr1
        data['scores'] = scores
        data['dsc0'] = dsc0
        data['dsc1'] = dsc1
        data['kp_scores'] = self.kp_matrix_scores(scr0, scr1)

        return kps0, dsc0, kps1, dsc1
