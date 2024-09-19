import torch
from lib.models.MicKey.modules.loss.solvers import weighted_procrustes
from lib.models.MicKey.modules.utils.training_utils import backproject_3d, soft_inlier_counting_3d, inlier_counting_3d

class e2eProbabilisticProcrustesSolver():
    """
    e2eProbabilisticProcrustesSolver computes the metric relative pose estimation during test time.
    Note that contrary to the training solver, here, the solver only refines the best pose hypothesis.
    Also, parameters are different during training and testing.
    """
    def __init__(self, cfg):

        # Populate Procrustes RANSAC parameters
        self.it_RANSAC = cfg.PROCRUSTES.IT_RANSAC
        self.it_matches = cfg.PROCRUSTES.IT_MATCHES
        self.num_samples_matches = cfg.PROCRUSTES.NUM_SAMPLED_MATCHES
        self.num_corr_3d_3d = cfg.PROCRUSTES.NUM_CORR_3D_3D
        self.num_refinements = cfg.PROCRUSTES.NUM_REFINEMENTS
        self.th_inlier = cfg.PROCRUSTES.TH_INLIER
        self.th_soft_inlier = cfg.PROCRUSTES.TH_SOFT_INLIER

    def estimate_pose(self, batch, return_inliers=False):
        '''
            Given 3D coordinates and matching matrices, estimate_pose computes the metric pose between query and reference images.
            args:
                return_inliers: Optional argument that indicates if a list of the inliers should be returned.
        '''

        matches = batch['final_scores'].detach()

        kps0, depth0 = batch['kps0'].detach(), batch['depth_kp0'].detach()
        kps1, depth1 = batch['kps1'].detach(), batch['depth_kp1'].detach()

        K0 = batch['K_color0'].float().detach()
        K1 = batch['K_color1'].float().detach()

        B, num_kpts, _ = matches.shape

        # Reshape to sample from matches matrix
        matches_row = matches.reshape(B, num_kpts*num_kpts)
        batch_idx = torch.tile(torch.arange(B).view(B, 1), [1, self.num_samples_matches]).reshape(B, self.num_samples_matches)
        batch_idx_ransac = torch.tile(torch.arange(B).view(B, 1), [1, self.num_corr_3d_3d]).reshape(B, self.num_corr_3d_3d)

        num_valid_h = 0
        Rs = torch.zeros((B, 0, 3, 3)).to(matches_row.device)
        ts = torch.zeros((B, 0, 1, 3)).to(matches_row.device)
        scores_ransac = torch.zeros((B, 0)).to(matches_row.device)

        # Keep track of X and Y correspondences subset
        it_matches_ids = []
        dict_corr = {}

        for i_i in range(self.it_matches):

            try:
                sampled_idx = torch.multinomial(matches_row, self.num_samples_matches)
            except:
                print('[Except Reached]: Invalid matching matrix! ')
                break

            sampled_idx_kp0 = torch.div(sampled_idx, num_kpts, rounding_mode='trunc')
            sampled_idx_kp1 = (sampled_idx % num_kpts)

            # Get kpt correspondences and discard the score - shape: (num_corr, 2)
            cor0 = kps0[batch_idx, :2, sampled_idx_kp0]
            cor1 = kps1[batch_idx, :2, sampled_idx_kp1]
            d0 = depth0[batch_idx, :2, sampled_idx_kp0]
            d1 = depth1[batch_idx, :2, sampled_idx_kp1]
            weights = matches_row[batch_idx, sampled_idx]

            X = backproject_3d(cor0, d0, K0)
            Y = backproject_3d(cor1, d1, K1)
            dict_corr[i_i] = {'X': X, 'Y': Y, 'cor0': cor0, 'cor1': cor1, 'd0': d0, 'd1': d1, 'weights': weights}

            for kk in range(self.it_RANSAC):

                sampled_idx_ransac = torch.multinomial(weights, self.num_corr_3d_3d)

                X_k = X[batch_idx_ransac, sampled_idx_ransac, :]
                Y_k = Y[batch_idx_ransac, sampled_idx_ransac, :]

                # get metric relative pose
                R, t, ok_rank = weighted_procrustes(X_k, Y_k, use_weights=False)

                if not ok_rank:
                    continue

                invalid_t = (torch.isnan(t).any() or torch.isinf(t).any())
                invalid_R = (torch.isnan(R).any() or torch.isinf(R).any())

                if invalid_t or invalid_R:
                    continue

                # Compute hypothesis score
                score_k = soft_inlier_counting_3d(X, Y, R, t, th=self.th_soft_inlier)

                Rs = torch.cat([Rs, R.unsqueeze(1)], 1)
                ts = torch.cat([ts, t.unsqueeze(1)], 1)
                scores_ransac = torch.cat([scores_ransac, score_k], 1)
                it_matches_ids.append(i_i)
                num_valid_h += 1

        if num_valid_h > 0:
            max_ind = torch.argmax(scores_ransac, dim=1)
            R = Rs[batch_idx_ransac[:, 0], max_ind]
            t_metric = ts[batch_idx_ransac[:, 0], max_ind]
            best_inliers = scores_ransac[batch_idx_ransac[:, 0], max_ind]

            # Use subset of correspondences that generated the hypothesis with maximum score
            X_best = torch.zeros_like(X)
            Y_best = torch.zeros_like(Y)
            for i_b in range(len(max_ind)):
                X_best[i_b], Y_best[i_b] = dict_corr[it_matches_ids[max_ind[i_b]]]['X'][i_b], dict_corr[it_matches_ids[max_ind[i_b]]]['Y'][i_b]
            inliers_ref = torch.zeros((B, self.num_samples_matches)).to(X.device)

            # 3D inliers:
            th_ref = self.num_refinements*[self.th_inlier]
            inliers_pre = self.num_corr_3d_3d * torch.ones_like(best_inliers)
            for i_ref in range(len(th_ref)):
                inliers = inlier_counting_3d(X_best, Y_best, R, t_metric, th=th_ref[i_ref])

                do_ref = (inliers.sum(-1) >= self.num_corr_3d_3d) * (inliers.sum(-1) > inliers_pre)
                inliers_pre[do_ref] = inliers.sum(-1)[do_ref]

                # Check whether any refinements need to be done
                if (do_ref.sum().float() == 0.).item():
                    break
                inliers_ref[do_ref] = inliers[do_ref]
                R[do_ref], t_metric[do_ref], _ = weighted_procrustes(X_best[do_ref], Y_best[do_ref],
                                                                     use_weights=True, use_mask=True,
                                                                     check_rank=False,
                                                                     w=inliers_ref[do_ref])

            best_inliers = soft_inlier_counting_3d(X_best, Y_best, R, t_metric, th=self.th_inlier)

        else:
            R = torch.zeros((B, 3, 3)).to(matches.device)
            t_metric = torch.zeros((B, 1, 3)).to(matches.device)
            best_inliers = torch.zeros((B)).to(matches.device)

        # Compute inliers set:
        if return_inliers:
            if num_valid_h > 0:

                # Use subset of correspondences that generated the hypothesis with maximum score
                cor0_best = torch.zeros_like(cor0)
                cor1_best = torch.zeros_like(cor1)
                X_best = torch.zeros_like(X)
                Y_best = torch.zeros_like(Y)
                d0_best = torch.zeros_like(d0)
                d1_best = torch.zeros_like(d1)
                weights_best = torch.zeros_like(weights)
                for i_b in range(len(max_ind)):
                    cor0_best[i_b], cor1_best[i_b] = dict_corr[it_matches_ids[max_ind[i_b]]]['cor0'][i_b], \
                                                     dict_corr[it_matches_ids[max_ind[i_b]]]['cor1'][i_b]
                    d0_best[i_b], d1_best[i_b] = dict_corr[it_matches_ids[max_ind[i_b]]]['d0'][i_b], \
                                                 dict_corr[it_matches_ids[max_ind[i_b]]]['d1'][i_b]
                    X_best[i_b], Y_best[i_b] = dict_corr[it_matches_ids[max_ind[i_b]]]['X'][i_b], \
                                               dict_corr[it_matches_ids[max_ind[i_b]]]['Y'][i_b]
                    weights_best[i_b] = dict_corr[it_matches_ids[max_ind[i_b]]]['weights'][i_b]

                # Compute inliers from latest sampled set of correspondences
                inliers_idxs = inlier_counting_3d(X_best, Y_best, R, t_metric, th=self.th_inlier)
                inliers = []
                for idx_b in range(len(inliers_idxs)):
                    cor0_inliers = cor0_best[idx_b, inliers_idxs[idx_b]==1.]
                    d0_inliers = d0_best[idx_b, inliers_idxs[idx_b]==1.]
                    cor1_inliers = cor1_best[idx_b, inliers_idxs[idx_b]==1.]
                    d1_inliers = d1_best[idx_b, inliers_idxs[idx_b]==1.]
                    score_inliers = weights_best[idx_b, inliers_idxs[idx_b]==1.]
                    order_corr = torch.argsort(score_inliers, descending=True)
                    inliers_b = torch.cat([cor0_inliers[order_corr], cor1_inliers[order_corr], score_inliers[order_corr].unsqueeze(-1), d0_inliers[order_corr], d1_inliers[order_corr]], dim=1)
                    inliers.append(inliers_b)
            else:
                inliers = [torch.zeros([0, 5])]*B

            return R, t_metric, best_inliers, inliers

        else:
            return R, t_metric, best_inliers


    def estimate_pose_vectorized(self, batch, return_inliers=False):
        '''
            Given 3D coordinates and matching matrices, estimate_pose computes the metric pose between query and reference images.
            Use a vectorized version of RANSAC instead of loops.
            args:
                return_inliers: Optional argument that indicates if a list of the inliers should be returned.
        '''

        matches = batch['final_scores'].detach()

        kps0, depth0 = batch['kps0'].detach(), batch['depth_kp0'].detach()
        kps1, depth1 = batch['kps1'].detach(), batch['depth_kp1'].detach()

        K0 = batch['K_color0'].float().detach()
        K1 = batch['K_color1'].float().detach()

        B, num_kpts, _ = matches.shape

        # Define the total number of candidate matches
        B, num_kpts, _ = matches.shape
        total_matches = num_kpts * num_kpts
        num_corr_3d_3d = self.num_corr_3d_3d

        # Define the new batch size due to RANSAC vectorization
        total_it = self.it_RANSAC * self.it_matches
        B_out_v = B * self.it_matches
        B_in_v = B * total_it

        # Reshape to sample from matches matrix
        matches_row = matches.reshape(B, num_kpts*num_kpts)

        num_valid_h = 0

        # Outler loop indices
        batch_idx = torch.arange(B)
        batch_idx_out_ransac_Bv = torch.tile(torch.arange(B).view(B, 1), [1, self.it_matches])
        batch_idx_out_ransac_Bv = torch.tile(batch_idx_out_ransac_Bv.view(B_out_v, 1), [1, self.num_samples_matches])

        # Inner loop indices
        batch_idx_in_ransac_Bv = torch.tile(torch.arange(B_in_v).view(B_in_v, 1), [1, num_corr_3d_3d]).reshape(B_in_v, num_corr_3d_3d)

        # Intrinsics for each iteration
        K0_out_v = torch.tile(K0.unsqueeze(1), [1, self.it_matches, 1, 1]).reshape(B_out_v, 3, 3)
        K1_out_v = torch.tile(K1.unsqueeze(1), [1, self.it_matches, 1, 1]).reshape(B_out_v, 3, 3)

        try:
            # Vectorized Outler RANSAC iterations
            matches_row_v = torch.tile(matches_row.unsqueeze(1), [1, self.it_matches, 1]).reshape(B_out_v, total_matches)
            sampled_idx_v = torch.multinomial(matches_row_v, self.num_samples_matches)

            sampled_idx_kp0 = torch.div(sampled_idx_v, num_kpts, rounding_mode='trunc')
            sampled_idx_kp1 = (sampled_idx_v % num_kpts)

            # Sample the positions according to the sample ids
            cor0 = kps0[batch_idx_out_ransac_Bv, :2, sampled_idx_kp0]
            cor1 = kps1[batch_idx_out_ransac_Bv, :2, sampled_idx_kp1]
            d0 = depth0[batch_idx_out_ransac_Bv, :2, sampled_idx_kp0]
            d1 = depth1[batch_idx_out_ransac_Bv, :2, sampled_idx_kp1]
            weights = matches_row[batch_idx_out_ransac_Bv, sampled_idx_v]

            X = backproject_3d(cor0, d0, K0_out_v)
            Y = backproject_3d(cor1, d1, K1_out_v)

            # Sample the 3D-3D correspondences
            X_v = torch.tile(X.unsqueeze(1), [1, self.it_RANSAC, 1, 1]).reshape(B_in_v, self.num_samples_matches, 3)
            Y_v = torch.tile(Y.unsqueeze(1), [1, self.it_RANSAC, 1, 1]).reshape(B_in_v, self.num_samples_matches, 3)
            weights_v = torch.tile(weights.unsqueeze(1), [1, self.it_RANSAC, 1]).reshape(B_in_v, self.num_samples_matches)

            sampled_idx_ransac_v = torch.multinomial(weights_v, self.num_corr_3d_3d)

            # Sample the 3D-3D correspondences
            X_k = X_v[batch_idx_in_ransac_Bv, sampled_idx_ransac_v, :]
            Y_k = Y_v[batch_idx_in_ransac_Bv, sampled_idx_ransac_v, :]
            weights_k = weights_v[batch_idx_in_ransac_Bv, sampled_idx_ransac_v]

            # get metric relative pose
            R, t, ok_rank = weighted_procrustes(X_k, Y_k, weights_k, use_weights=False, check_rank=False)

            invalid_t = (torch.isnan(t).any() or torch.isinf(t).any())
            invalid_R = (torch.isnan(R).any() or torch.isinf(R).any())

            # Compute hypothesis score
            score_k = soft_inlier_counting_3d(X_v, Y_v, R, t, th=self.th_soft_inlier)

            # Reshape poses back to RANSAC iteration shape
            score_k = score_k.reshape(B, total_it)
            R = R.reshape(B, total_it, 3, 3)
            t = t.reshape(B, total_it, 1, 3)
            X_v = X_v.reshape(B, total_it, 2048, 3)
            Y_v = Y_v.reshape(B, total_it, 2048, 3)

            # Find best hypothesis RANSAC
            max_RANSAC_it = torch.argmax(score_k, dim=1)
            R = R[batch_idx, max_RANSAC_it]
            t = t[batch_idx, max_RANSAC_it]
            best_score = score_k[batch_idx, max_RANSAC_it]

            X_best = X_v[batch_idx, max_RANSAC_it]
            Y_best = Y_v[batch_idx, max_RANSAC_it]
            inliers_ref = torch.zeros((B, self.num_samples_matches)).to(X.device)

            # 3D inliers:
            th_ref = self.num_refinements * [self.th_inlier]
            inliers_pre = self.num_corr_3d_3d * torch.ones_like(best_score)
            for i_ref in range(len(th_ref)):
                inliers = inlier_counting_3d(X_best, Y_best, R, t, th=th_ref[i_ref])

                do_ref = (inliers.sum(-1) >= self.num_corr_3d_3d) * (inliers.sum(-1) > inliers_pre)
                inliers_pre[do_ref] = inliers.sum(-1)[do_ref]

                # Check whether any refinements need to be done
                if (do_ref.sum().float() == 0.).item():
                    break
                inliers_ref[do_ref] = inliers[do_ref]
                R[do_ref], t[do_ref], _ = weighted_procrustes(X_best[do_ref], Y_best[do_ref],
                                                                     use_weights=True, use_mask=True,
                                                                     check_rank=False,
                                                                     w=inliers_ref[do_ref])

            # Compute final pose score:
            best_inliers = soft_inlier_counting_3d(X_best, Y_best, R, t, th=self.th_inlier)

            inliers = [torch.zeros([0, 5])] * B
            if return_inliers:
                # Compute inliers from latest sampled set of correspondences
                inliers_idxs = inlier_counting_3d(X_best, Y_best, R, t, th=self.th_inlier)
                cor0 = cor0.reshape(B, self.it_matches, self.num_samples_matches, 2)[batch_idx, max_RANSAC_it//self.it_RANSAC]
                d0 = d0.reshape(B, self.it_matches, self.num_samples_matches, 1)[batch_idx, max_RANSAC_it//self.it_RANSAC]
                cor1 = cor1.reshape(B, self.it_matches, self.num_samples_matches, 2)[batch_idx, max_RANSAC_it//self.it_RANSAC]
                d1 = d1.reshape(B, self.it_matches, self.num_samples_matches, 1)[batch_idx, max_RANSAC_it//self.it_RANSAC]
                weights = weights.reshape(B, self.it_matches, self.num_samples_matches)[batch_idx, max_RANSAC_it//self.it_RANSAC]
                inliers = []
                for idx_b in range(len(inliers_idxs)):

                    cor0_inliers = cor0[idx_b, inliers_idxs[idx_b] == 1.]
                    d0_inliers = d0[idx_b, inliers_idxs[idx_b] == 1.]
                    cor1_inliers = cor1[idx_b, inliers_idxs[idx_b] == 1.]
                    d1_inliers = d1[idx_b, inliers_idxs[idx_b] == 1.]
                    score_inliers = weights[idx_b, inliers_idxs[idx_b] == 1.]

                    order_corr = torch.argsort(score_inliers, descending=True)
                    inliers_b = torch.cat(
                        [cor0_inliers[order_corr], cor1_inliers[order_corr], score_inliers[order_corr].unsqueeze(-1),
                         d0_inliers[order_corr], d1_inliers[order_corr]], dim=1)
                    inliers.append(inliers_b)

            if not (invalid_t or invalid_R):
                num_valid_h += 1
        except:
            print('[Except Reached]: Invalid Procrustes configuration! ')
            R = torch.zeros((B, 3, 3)).to(matches.device)
            t = torch.zeros((B, 1, 3)).to(matches.device)
            best_inliers = torch.zeros((B)).to(matches.device)
            inliers = [torch.zeros([0, 5])] * B

        if num_valid_h == 0:
            R = torch.zeros((B, 3, 3)).to(matches.device)
            t = torch.zeros((B, 1, 3)).to(matches.device)
            best_inliers = torch.zeros((B)).to(matches.device)
            inliers = [torch.zeros([0, 5])] * B

        # Compute inliers set:
        if return_inliers:
            return R, t, best_inliers, inliers
        else:
            return R, t, best_inliers
