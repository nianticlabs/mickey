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
                inliers_idxs = inlier_counting_3d(X_best, Y_best, R, t_metric, th=0.15)
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

