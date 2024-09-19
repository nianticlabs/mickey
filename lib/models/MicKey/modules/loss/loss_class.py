import torch
import torch.nn as nn
import numpy as np

from lib.models.MicKey.modules.utils.training_utils import backproject_3d, soft_inlier_counting_3d, inlier_counting_3d
from lib.models.MicKey.modules.loss.loss_utils import compute_pose_loss, compute_vcre_loss
from lib.models.MicKey.modules.loss.solvers import weighted_procrustes

class MetricPoseLoss(nn.Module):
    """
    MetricPoseLoss computes the metric relative pose estimation during training time.
    MetricPoseLoss computes the gradients for the REINFORCE algorithm as well as providing a direct signal
    to learn the 3D keypoint coordiantes, ie, 2D offsets + depths.
    """
    def __init__(self, cfg):
        super().__init__()

        # Defien the loss parameters
        self.define_loss_function(cfg)
        self.populate_loss_parameters(cfg)

    def define_loss_function(self, cfg):

        # Loss type: Supported VCRE or POSE_ERR
        self.loss_type = cfg.LOSS_CLASS.LOSS_FUNCTION
        self.soft_clipping = cfg.LOSS_CLASS.SOFT_CLIPPING

        if self.loss_type == 'POSE_ERR':
            self.compute_loss = compute_pose_loss
            if self.soft_clipping:
                self.max_loss_null = cfg.LOSS_CLASS.POSE_ERR.MAX_LOSS_SOFTVALUE
            else:
                self.max_loss_null = cfg.LOSS_CLASS.POSE_ERR.MAX_LOSS_VALUE
        elif self.loss_type == 'VCRE':
            self.compute_loss = compute_vcre_loss
            if self.soft_clipping:
                self.max_loss_null = cfg.LOSS_CLASS.VCRE.MAX_LOSS_SOFTVALUE
            else:
                self.max_loss_null = cfg.LOSS_CLASS.VCRE.MAX_LOSS_VALUE

    def populate_loss_parameters(self, cfg):

        self.num_samples_matches = cfg.LOSS_CLASS.SAMPLER.NUM_SAMPLES_MATCHES

        # Use RANSAC Vectorized version for faster training
        self.use_RANSAC_vectorized = True

        # Robust fitting:
        self.score_temperature = cfg.LOSS_CLASS.GENERATE_HYPOTHESES.SCORE_TEMPERATURE
        self.it_matches = cfg.LOSS_CLASS.GENERATE_HYPOTHESES.IT_MATCHES
        self.it_RANSAC = cfg.LOSS_CLASS.GENERATE_HYPOTHESES.IT_RANSAC
        self.inlier_3d_th = cfg.LOSS_CLASS.GENERATE_HYPOTHESES.INLIER_3D_TH
        self.inlier_ref_th = cfg.LOSS_CLASS.GENERATE_HYPOTHESES.INLIER_REF_TH
        self.num_ref_steps = cfg.LOSS_CLASS.GENERATE_HYPOTHESES.NUM_REF_STEPS
        self.num_corr_3d_3d = cfg.LOSS_CLASS.GENERATE_HYPOTHESES.NUM_CORR_3d3d

        # Null Hypothesis
        self.add_null_hypothesis = cfg.LOSS_CLASS.NULL_HYPOTHESIS.ADD_NULL_HYPOTHESIS
        self.th_outliers = cfg.LOSS_CLASS.NULL_HYPOTHESIS.TH_OUTLIERS

        # Curriculum learning
        self.train_w_top = cfg.LOSS_CLASS.CURRICULUM_LEARNING.TRAIN_WITH_TOPK or \
                           cfg.LOSS_CLASS.CURRICULUM_LEARNING.TRAIN_CURRICULUM
        if cfg.LOSS_CLASS.CURRICULUM_LEARNING.TRAIN_CURRICULUM:
            self.topK = cfg.LOSS_CLASS.CURRICULUM_LEARNING.TOPK_INIT
        elif cfg.LOSS_CLASS.CURRICULUM_LEARNING.TRAIN_WITH_TOPK:
            self.topK = cfg.LOSS_CLASS.CURRICULUM_LEARNING.TOPK

    def read_pose_parameters(self, batch):
        # Auxiliary funciton to read the pose and intrinsic parameters

        Rgt = batch['T_0to1'][:, :3, :3]
        tgt = batch['T_0to1'][:, :3, 3:].transpose(1, 2)
        K0 = batch['K_color0'].float()
        K1 = batch['K_color1'].float()

        return Rgt, tgt, K0, K1

    def single_iteration_RANSAC(self, batch, check_rank):

        # Detach output of the network and accumulate gradients (later use for the direct signal, ie, 3D coordinates)
        matches = batch['final_scores'].detach()
        kps0, depth0 = batch['kps0'].detach().requires_grad_(), batch['depth_kp0'].detach().requires_grad_()
        kps1, depth1 = batch['kps1'].detach().requires_grad_(), batch['depth_kp1'].detach().requires_grad_()

        # Define the total number of candidate matches
        B, num_kpts, _ = matches.shape
        total_matches = num_kpts * num_kpts
        num_corr_3d_3d = self.num_corr_3d_3d

        # Define the new batch size due to RANSAC vectorization
        total_it = self.it_RANSAC * self.it_matches
        B_out_v = B * self.it_matches
        B_in_v = B * total_it

        # Reshape to sample from matches matrix
        matches_row = matches.reshape(B, total_matches)

        # Outler loop indices
        batch_idx_out_ransac_Bv = torch.tile(torch.arange(B).view(B, 1), [1, self.it_matches])
        batch_idx_out_ransac_Bv = torch.tile(batch_idx_out_ransac_Bv.view(B_out_v, 1), [1, self.num_samples_matches])

        # Inner loop indices
        batch_idx_in_ransac_Bv = torch.tile(torch.arange(B_in_v).view(B_in_v, 1), [1, num_corr_3d_3d]).reshape(B_in_v, num_corr_3d_3d)

        # Define ground truth pose parameters
        Rgt, tgt, K0, K1 = self.read_pose_parameters(batch)

        Rgt_v = torch.tile(Rgt.unsqueeze(1), [1, self.it_RANSAC*self.it_matches, 1, 1]).reshape(B_in_v, 3, 3)
        tgt_v = torch.tile(tgt.unsqueeze(1), [1, self.it_RANSAC*self.it_matches, 1, 1]).reshape(B_in_v, 1, 3)
        K0_out_v = torch.tile(K0.unsqueeze(1), [1, self.it_matches, 1, 1]).reshape(B_out_v, 3, 3)
        K1_out_v = torch.tile(K1.unsqueeze(1), [1, self.it_matches, 1, 1]).reshape(B_out_v, 3, 3)
        K0ori_in_v = torch.tile(batch['Kori_color0'].unsqueeze(1), [1, total_it, 1, 1]).reshape(B_in_v, 3, 3)
        K1ori_in_v = torch.tile(batch['Kori_color1'].unsqueeze(1), [1, total_it, 1, 1]).reshape(B_in_v, 3, 3)

        # Auxiliary variables to accumulate losses
        num_valid_h = 0
        baseline = torch.zeros((B,)).to(matches_row.device)
        losses_rot = torch.zeros((B, 1)).to(matches_row.device)
        losses_trans = torch.zeros((B, 1)).to(matches_row.device)

        # this tensor will contain the gradients for the entire batch
        gradients = torch.zeros_like(matches_row)
        gradients_b = torch.zeros_like(matches_row)

        # Check if matching matrix has any invalid value
        invalid_matches = (torch.isnan(matches_row).any() or torch.isinf(matches_row).any() or
                           (matches_row < 0.).any() or (matches_row.sum(-1) < 0.).any() or (
                                       matches_row.sum() < 0.).any())

        if not invalid_matches:

            # Use matching probabilities to guide the sampling step
            try:
                # Vectorized Outler RANSAC iterations
                matches_row_v = torch.tile(matches_row.unsqueeze(1), [1, self.it_matches, 1]).reshape(B_out_v,
                                                                                                      total_matches)
                sampled_idx_v = torch.multinomial(matches_row_v, self.num_samples_matches)

                sampled_idx_kp0 = torch.div(sampled_idx_v, num_kpts, rounding_mode='trunc')
                sampled_idx_kp1 = (sampled_idx_v % num_kpts)

                # Sample the positions according to the sample ids
                cor0 = kps0[batch_idx_out_ransac_Bv, :2, sampled_idx_kp0]
                cor1 = kps1[batch_idx_out_ransac_Bv, :2, sampled_idx_kp1]
                d0 = depth0[batch_idx_out_ransac_Bv, :2, sampled_idx_kp0]
                d1 = depth1[batch_idx_out_ransac_Bv, :2, sampled_idx_kp1]
                weights = matches_row[batch_idx_out_ransac_Bv, sampled_idx_v]

                # Project to camera space
                X = backproject_3d(cor0, d0, K0_out_v)
                Y = backproject_3d(cor1, d1, K1_out_v)

                # Vectorized RANSAC iterations
                X_v = torch.tile(X.unsqueeze(1), [1, self.it_RANSAC, 1, 1]).reshape(B_in_v, self.num_samples_matches, 3)
                Y_v = torch.tile(Y.unsqueeze(1), [1, self.it_RANSAC, 1, 1]).reshape(B_in_v, self.num_samples_matches, 3)
                weights_v = torch.tile(weights.unsqueeze(1), [1, self.it_RANSAC, 1]).reshape(B_in_v, self.num_samples_matches)

                sampled_idx_ransac_v = torch.multinomial(weights_v, num_corr_3d_3d)

                # Start refinement process
                # Avoid gradients through refinement
                with torch.no_grad():
                    # Compute first the mask for inlier correspondences - Use more restrictive th each iteration?
                    th_ref = self.num_ref_steps * [self.inlier_ref_th]
                    inliers_pre = self.num_corr_3d_3d * torch.ones((B_in_v,)).to(X.device)
                    inliers_ref = torch.zeros((B_in_v, self.num_samples_matches)).to(X.device)
                    inliers_final = torch.zeros((B_in_v, self.num_samples_matches)).to(X.device)
                    inliers_final[batch_idx_in_ransac_Bv, sampled_idx_ransac_v] = 1
                    inliers = torch.zeros((B_in_v, self.num_samples_matches)).to(X.device)
                    inliers[batch_idx_in_ransac_Bv, sampled_idx_ransac_v] = 1
                    do_ref = torch.ones((B_in_v)).to(X.device).bool()
                    R_detach = torch.zeros((B_in_v, 3, 3)).to(X.device)
                    t_detach = torch.zeros((B_in_v, 1, 3)).to(X.device)

                    # Refinement loop
                    for i_ref in range(len(th_ref)):

                        # Use all inliers to compute the new pose
                        R_detach[do_ref], t_detach[do_ref], _ = weighted_procrustes(X_v[do_ref], Y_v[do_ref],
                                                                                    use_mask=True,
                                                                                    use_weights=True, check_rank=False,
                                                                                    w=inliers[do_ref])

                        # Find the 3D inliers for the candidate relative pose
                        inliers_ref[do_ref] = inlier_counting_3d(X_v[do_ref], Y_v[do_ref], R_detach[do_ref],
                                                                 t_detach[do_ref], th=th_ref[i_ref])

                        do_ref = inliers_ref.sum(-1) > inliers_pre
                        inliers_pre[do_ref] = inliers_ref.sum(-1)[do_ref]
                        inliers_final[do_ref] = inliers[do_ref]
                        inliers[do_ref] = inliers_ref[do_ref]

                        # Check whether any refinements need to be done
                        if (do_ref.sum().float() == 0.).item():
                            break

                # Recompute the pose with the final set of inliers. This time, gradients are accumulated
                R, t, ok_rank = weighted_procrustes(X_v, Y_v, use_weights=True, use_mask=True, w=inliers_final,
                                                    check_rank=check_rank)

                # Check whether the relative pose is generated correctly
                if not ok_rank:
                    print('[ERROR]: Skipping RANSAC iteration due to rank matrix.')
                    outputs = {}
                    outputs['kps0'] = kps0
                    outputs['kps1'] = kps1
                    outputs['depth0'] = depth0
                    outputs['depth1'] = depth1
                    return baseline, losses_rot, losses_trans, gradients, gradients_b, outputs, num_valid_h

                # Skip any invalid pose
                invalid_t = (torch.isnan(t).any() or torch.isinf(t).any())
                invalid_R = (torch.isnan(R).any() or torch.isinf(R).any())

                if invalid_t or invalid_R:
                    print('[ERROR]: Skipping RANSAC iteration due to invalid values in R/t.')
                    outputs = {}
                    outputs['kps0'] = kps0
                    outputs['kps1'] = kps1
                    outputs['depth0'] = depth0
                    outputs['depth1'] = depth1
                    return baseline, losses_rot, losses_trans, gradients, gradients_b, outputs, num_valid_h

                # Compute score with refined pose
                score_k = soft_inlier_counting_3d(X_v, Y_v, R, t, th=self.inlier_3d_th)

                # Compute the loss of the estimated pose wrt to ground truth
                loss_value_k, loss_rot_k, loss_trans_k = self.compute_loss(R, t, Rgt_v, tgt_v, K0ori_in_v, K1ori_in_v, soft_clipping=self.soft_clipping)

                # Reshape losses back to RANSAC iteration shape
                loss_value_k = loss_value_k.reshape(B_out_v, self.it_RANSAC)
                loss_rot_k = loss_rot_k.reshape(B_out_v, self.it_RANSAC)
                loss_trans_k = loss_trans_k.reshape(B_out_v, self.it_RANSAC)
                score_k = score_k.reshape(B_out_v, self.it_RANSAC)

                # Aggregate the losses based on their soft inlier counting (score)
                loss_rot = (loss_rot_k * torch.softmax(score_k / self.score_temperature, -1)).sum(-1).unsqueeze(-1)
                loss_trans = (loss_trans_k * torch.softmax(score_k / self.score_temperature, -1)).sum(-1).unsqueeze(-1)

                if self.add_null_hypothesis:
                    null_score = (self.th_outliers * self.num_samples_matches) * torch.ones((B_out_v, 1)).to(kps0.device)
                    null_loss = self.max_loss_null * torch.ones((B_out_v, 1)).to(kps0.device)
                    loss_value_k = torch.cat([loss_value_k, null_loss], -1)
                    score_k = torch.cat([score_k, null_score], -1)

                # Compute the final loss value for this REINFORCE iteration
                loss_value = (loss_value_k * torch.softmax(score_k/self.score_temperature, -1)).sum(-1).unsqueeze(-1)

                # Store gradients in a for loop (in case once position has been sampled more than once)
                for i_it in range(B_out_v):

                    # Aggreagate the selected positions for every RANSAC iteration:
                    gradients_b[batch_idx_out_ransac_Bv[i_it].unsqueeze(0), sampled_idx_v[i_it].unsqueeze(0)] += 1

                    # Aggregate the losses for every RANSAC iteration:
                    gradients_tmp = torch.zeros_like(matches_row)
                    gradients_tmp[batch_idx_out_ransac_Bv[i_it].unsqueeze(0), sampled_idx_v[i_it].unsqueeze(0)] += 1
                    gradients_tmp[batch_idx_out_ransac_Bv[i_it].unsqueeze(0), sampled_idx_v[i_it].unsqueeze(0)] *= loss_value[i_it]

                    gradients += gradients_tmp

                losses_rot = loss_rot.reshape(B, self.it_matches).sum(-1).unsqueeze(-1)
                losses_trans = loss_trans.reshape(B, self.it_matches).sum(-1).unsqueeze(-1)
                baseline = loss_value.reshape(B, self.it_matches).sum(-1)

                num_valid_h += 1

            except:
                print('[Except Reached]: Invalid matching matrix! ')
                outputs = {}
                outputs['kps0'] = kps0
                outputs['kps1'] = kps1
                outputs['depth0'] = depth0
                outputs['depth1'] = depth1
                return baseline, losses_rot, losses_trans, gradients, gradients_b, outputs, num_valid_h
        else:
            print('Invalid matching matrix! Skip RANSAC loop.')

        outputs = {}
        outputs['kps0'] = kps0
        outputs['kps1'] = kps1
        outputs['depth0'] = depth0
        outputs['depth1'] = depth1
        return baseline, losses_rot, losses_trans, gradients, gradients_b, outputs, num_valid_h

    def RANSAC_vectorized(self, batch, check_rank=False):

        B, num_kpts, _ = batch['final_scores'].shape
        baseline, losses_rot, losses_trans, gradients, gradients_b, outputs, num_valid_h = \
            self.single_iteration_RANSAC(batch, check_rank)

        # calculate the gradients of the expected loss
        baseline = baseline / self.it_matches  # expected loss
        losses_trans = losses_trans / self.it_matches  # expected rot loss
        losses_rot = losses_rot / self.it_matches  # expected trans loss

        # Aggregate the gradients
        gradients = gradients - (gradients_b * baseline.view(B, 1))

        # Divide based on total number of iterations: Give more weight to more sampled correspondence candidates
        gradients = gradients / self.it_matches

        # Check if no valid poses were generated
        if num_valid_h == 0:
            print('[ERROR]: No valid hypotheses generated!')

        # If curriculum learning select the top K image pairs
        if self.train_w_top and B > 1:
            select_topB = np.maximum(int(B * self.topK / 100), 1)

            topk_loss = baseline[torch.argsort(baseline)[select_topB]]
            mask_topk = (baseline < topk_loss).float()

            avg_loss = (mask_topk * baseline).sum() / mask_topk.sum()
            gradients = gradients * mask_topk.unsqueeze(-1)
        else:
            avg_loss = torch.mean(baseline)
            mask_topk = torch.ones(B).to(avg_loss.device)

        # Store metrics:
        outputs['avg_loss_rot'] = torch.mean(losses_rot)
        outputs['avg_loss_trans'] = torch.mean(losses_trans)
        outputs['avg_rot_errs'] = torch.mean(torch.rad2deg(torch.as_tensor(losses_rot)))
        outputs['avg_t_errs'] = torch.mean(losses_trans)
        outputs['mask_topk'] = mask_topk
        gradients = gradients.reshape(B, num_kpts, num_kpts)

        return avg_loss, outputs, [gradients], num_valid_h

    def RANSAC_loops(self, batch):
        # Detach output of the network and accumulate gradients (later use for the direct signal, ie, 3D coordinates)
        matches = batch['final_scores'].detach()
        kps0, depth0 = batch['kps0'].detach().requires_grad_(), batch['depth_kp0'].detach().requires_grad_()
        kps1, depth1 = batch['kps1'].detach().requires_grad_(), batch['depth_kp1'].detach().requires_grad_()

        # Define ground truth pose parameters
        Rgt, tgt, K0, K1 = self.read_pose_parameters(batch)

        # Define the total number of candidate matches
        B, num_kpts, _ = matches.shape
        num_corr_3d_3d = self.num_corr_3d_3d

        # Reshape to sample from matches matrix
        matches_row = matches.reshape(B, num_kpts*num_kpts)
        batch_idx = torch.tile(torch.arange(B).view(B, 1), [1, self.num_samples_matches]).reshape(B, self.num_samples_matches)
        batch_idx_ransac = torch.tile(torch.arange(B).view(B, 1), [1, num_corr_3d_3d]).reshape(B, num_corr_3d_3d)

        # Auxiliary variables to accumulate losses
        losses = []
        num_valid_h = 0
        baseline = torch.zeros((B,)).to(matches_row.device)
        losses_rot = torch.zeros((B, 1)).to(matches_row.device)
        losses_trans = torch.zeros((B, 1)).to(matches_row.device)

        # this tensor will contain the gradients for the entire batch
        gradients = torch.zeros_like(matches_row)
        gradients_b = torch.zeros_like(matches_row)

        # REINFORCE loop
        for i_i in range(self.it_matches):

            # Variable that accumulates the gradients of this iteration only
            gradients_tmp = torch.zeros_like(matches_row)

            # Check if matching matrix has any invalid value
            invalid_matches = (torch.isnan(matches_row).any() or torch.isinf(matches_row).any() or
                               (matches_row < 0.).any() or (matches_row.sum(-1) < 0.).any() or (matches_row.sum() < 0.).any())
            if invalid_matches:
                print('Invalid matching matrix! ')
                break

            # Use matching probabilities to guide the sampling step
            try:
                sampled_idx = torch.multinomial(matches_row, self.num_samples_matches)
            except:
                print('[Except Reached]: Invalid matching matrix! ')
                break

            sampled_idx_kp0 = torch.div(sampled_idx, num_kpts, rounding_mode='trunc')
            sampled_idx_kp1 = (sampled_idx % num_kpts)

            # Sample the positions according to the sample ids
            cor0 = kps0[batch_idx, :2, sampled_idx_kp0]
            cor1 = kps1[batch_idx, :2, sampled_idx_kp1]
            d0 = depth0[batch_idx, :2, sampled_idx_kp0]
            d1 = depth1[batch_idx, :2, sampled_idx_kp1]
            weights = matches_row[batch_idx, sampled_idx]

            # Project to camera space
            X = backproject_3d(cor0, d0, K0)
            Y = backproject_3d(cor1, d1, K1)

            # Variables that accumulate the scores/losses over the RANSAC loop
            scores_ransac = torch.zeros((B, 0)).to(kps0.device)
            losses_ransac = torch.zeros((B, 0)).to(kps0.device)
            losses_R_ransac = torch.zeros((B, 0)).to(kps0.device)
            losses_t_ransac = torch.zeros((B, 0)).to(kps0.device)

            # RANSAC loop
            for kk in range(self.it_RANSAC):

                try:
                    sampled_idx_ransac = torch.multinomial(weights, num_corr_3d_3d)
                except:
                    print('[Except Reached]: Invalid matching matrix: torch.multinomial(weights, num_corr_3d_3d)')
                    break

                # Sample the 3D-3D correspondences
                X_k = X[batch_idx_ransac, sampled_idx_ransac, :]
                Y_k = Y[batch_idx_ransac, sampled_idx_ransac, :]
                weights_k = weights[batch_idx_ransac, sampled_idx_ransac]

                # get metric relative pose
                R_pre, t_pre, ok_rank = weighted_procrustes(X_k, Y_k, weights_k, use_weights=False)

                # Check whether the relative pose is generated correctly
                if not ok_rank:
                    continue

                # Skip any invalid pose
                invalid_t = (torch.isnan(t_pre).any() or torch.isinf(t_pre).any())
                invalid_R = (torch.isnan(R_pre).any() or torch.isinf(R_pre).any())

                if invalid_t or invalid_R:
                    continue

                # Compute hypothesis score
                score_k = soft_inlier_counting_3d(X, Y, R_pre, t_pre, th=self.inlier_3d_th)

                # Start refinement process
                # Avoid gradients through refinement
                with torch.no_grad():
                    # Compute first the mask for inlier correspondences - Use more restrictive th each iteration?
                    th_ref = self.num_ref_steps * [self.inlier_ref_th]
                    inliers_pre = torch.zeros((B,)).to(X.device)
                    inliers_ref = torch.zeros((B, self.num_samples_matches)).to(X.device)
                    inliers_ref[batch_idx_ransac, sampled_idx_ransac] = 1
                    R_detach, t_detach = R_pre.clone().detach(), t_pre.clone().detach()

                    # Refinement loop
                    for i_ref in range(len(th_ref)):

                        # Find the 3D inliers for the candidate relative pose
                        inliers = inlier_counting_3d(X, Y, R_detach, t_detach, th=th_ref[i_ref])
                        do_ref = (inliers.sum(-1) >= self.num_corr_3d_3d) * (inliers.sum(-1) > inliers_pre)
                        inliers_pre = inliers.sum(-1)

                        # Check whether any refinements need to be done
                        if (do_ref.sum().float() == 0.).item():
                            break

                        # Use all inliers to compute the new pose
                        inliers_ref[do_ref] = inliers[do_ref]
                        R_detach[do_ref], t_detach[do_ref], _ = weighted_procrustes(X[do_ref], Y[do_ref], use_mask=True,
                                                                              use_weights=True, check_rank = False,
                                                                              w=inliers_ref[do_ref])

                # Recompute the pose with the final set of inliers. This time, gradients are accumulated
                R, t, ok_rank = weighted_procrustes(X, Y, use_weights=True, use_mask=True, w=inliers_ref)

                # Check whether the relative pose is generated correctly
                if not ok_rank:
                    continue

                # Skip any invalid pose
                invalid_t = (torch.isnan(t_pre).any() or torch.isinf(t_pre).any())
                invalid_R = (torch.isnan(R_pre).any() or torch.isinf(R_pre).any())

                if invalid_t or invalid_R:
                    continue

                # Compute the loss of the estimated pose wrt to ground truth
                loss_value_k, loss_rot_k, loss_trans_k = self.compute_loss(R, t, Rgt, tgt, batch['Kori_color0'], soft_clipping=self.soft_clipping)

                # Accumulate the loss values
                losses_ransac = torch.cat([losses_ransac, loss_value_k], -1)
                losses_R_ransac = torch.cat([losses_R_ransac, loss_rot_k], -1)
                losses_t_ransac = torch.cat([losses_t_ransac, loss_trans_k], -1)
                scores_ransac = torch.cat([scores_ransac, score_k], -1)

            # Aggregate the losses based on their soft inlier counting (score)
            if scores_ransac.shape[1] > 0:
                loss_rot = (losses_R_ransac * torch.softmax(scores_ransac / self.score_temperature, -1)).sum(-1).unsqueeze(-1)
                loss_trans = (losses_t_ransac * torch.softmax(scores_ransac / self.score_temperature, -1)).sum(-1).unsqueeze(-1)

                if self.add_null_hypothesis:
                    null_score = (self.th_outliers * self.num_samples_matches) * torch.ones((B, 1)).to(kps0.device)
                    null_loss = self.max_loss_null * torch.ones((B, 1)).to(kps0.device)
                    losses_ransac = torch.cat([losses_ransac, null_loss], -1)
                    scores_ransac = torch.cat([scores_ransac, null_score], -1)

                # Compute the final loss value for this REINFORCE iteration
                loss_value = (losses_ransac * torch.softmax(scores_ransac/self.score_temperature, -1)).sum(-1).unsqueeze(-1)

                # gradient tensor of the current REINFORCE iteration
                gradients_b[batch_idx, sampled_idx] += 1

                gradients_tmp[batch_idx, sampled_idx] += 1
                gradients_tmp[batch_idx, sampled_idx] *= loss_value
                gradients += gradients_tmp

                losses.append(loss_value)
                losses_rot += loss_rot
                losses_trans += loss_trans
                baseline += loss_value[:, 0]

                num_valid_h += 1

        # calculate the gradients of the expected loss
        baseline = baseline / len(losses)  # expected loss
        losses_trans = losses_trans / len(losses)  # expected rot loss
        losses_rot = losses_rot / len(losses)  # expected trans loss

        # Aggregate the gradients
        gradients = gradients - (gradients_b * baseline.view(B, 1))

        # Divide based on total number of iterations: Give more weight to more sampled correspondence candidates
        gradients = gradients / np.maximum(len(losses), 1)

        # Check if no valid poses were generated
        if num_valid_h == 0:
            print('[ERROR]: No valid hypotheses generated!')

        # If curriculum learning select the top K image pairs
        if self.train_w_top and B>1:
            select_topB =  np.maximum(int(B * self.topK / 100), 1)

            topk_loss = baseline[torch.argsort(baseline)[select_topB]]
            mask_topk = (baseline < topk_loss).float()

            avg_loss = (mask_topk * baseline).sum() / mask_topk.sum()
            gradients = gradients * mask_topk.unsqueeze(-1)
        else:
            avg_loss = torch.mean(baseline)
            mask_topk = torch.ones(B).to(avg_loss.device)

        # Store metrics:
        outputs = {}
        outputs['avg_loss_rot'] = torch.mean(losses_rot)
        outputs['avg_loss_trans'] = torch.mean(losses_trans)
        outputs['avg_rot_errs'] = torch.mean(torch.rad2deg(torch.as_tensor(losses_rot)))
        outputs['avg_t_errs'] = torch.mean(losses_trans)
        outputs['kps0'] = kps0
        outputs['kps1'] = kps1
        outputs['depth0'] = depth0
        outputs['depth1'] = depth1
        outputs['mask_topk'] = mask_topk
        gradients = gradients.reshape(B, num_kpts, num_kpts)

        return avg_loss, outputs, [gradients], num_valid_h

    def forward(self, batch):

        if self.use_RANSAC_vectorized:
            return self.RANSAC_vectorized(batch)
        else:
            return self.RANSAC_loops(batch)
