import torch
import torch.nn as nn
import torch.nn.functional as F

def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1

class featureMatcher(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        if cfg['TYPE'] == 'DualSoftmax':
            self.matching_mat = dualSoftmax(cfg['DUAL_SOFTMAX'])
        elif cfg['TYPE'] == 'Sinkhorn':
            self.matching_mat = sinkhorn(cfg['SINKHORN'])
        else:
            print('[ERROR]: feature matcher not recognized')

    def get_matches_list(self, scores, min_conf=0.0):

        # Supports batch_size = 1

        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        # zero = scores.new_tensor(0)
        zero = torch.tensor(0).to(scores.device).float()
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        valid0 = mutual0 & (mscores0 > min_conf)
        # indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        minus_one = torch.tensor(-1).to(scores.device)
        indices0 = torch.where(valid0, indices0, minus_one)

        valid = indices0 > -1

        idx0 = arange_like(indices0, 1)[valid[0]].unsqueeze(1)
        idx1 = indices0[valid].unsqueeze(1)

        matches = torch.concat([idx0, idx1], dim=1)

        batch_idx = torch.tile(torch.arange(1).view(1, 1), [1, len(matches)]).reshape(-1)
        scores_matches = scores[batch_idx, idx0[:, 0], idx1[:, 0]]
        _, idx_sorted = torch.sort(scores_matches, descending=True)

        return matches[idx_sorted]

    def forward(self, kpt0, dsc0, kpt1, dsc1):

        scores = self.matching_mat(dsc0, dsc1)

        return scores

class dualSoftmax(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.temperature = cfg['TEMPERATURE']
        self.use_dustbin = False
        if cfg['USE_DUSTBIN']:
            self.dustbin_score = nn.Parameter(torch.tensor(1.))
            self.use_dustbin = True

    def forward(self, dsc0, dsc1):
        scores = torch.matmul(dsc0.transpose(1, 2).contiguous(), dsc1) / self.temperature

        if self.use_dustbin:
            b, m, n = scores.shape

            bins0 = self.dustbin_score.expand(b, m, 1)
            bins1 = self.dustbin_score.expand(b, 1, n)
            alpha = self.dustbin_score.expand(b, 1, 1)

            couplings = torch.cat([torch.cat([scores, bins0], -1),
                                   torch.cat([bins1, alpha], -1)], 1)

            couplings = F.softmax(couplings, 1) * F.softmax(couplings, 2)
            scores = couplings[:, :-1, :-1]

        else:
            scores = F.softmax(scores, 1) * F.softmax(scores, 2)

        return scores

class sinkhorn(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.dustbin_score = nn.Parameter(torch.tensor(cfg['DUSTBIN_SCORE_INIT']))
        self.sinkhorn_iterations = cfg['NUM_IT']
        self.descriptor_dim = 128

    def log_sinkhorn_iterations(self, Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor,
                                iters: int) -> torch.Tensor:
        """ Perform Sinkhorn Normalization in Log-space for stability"""
        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
        for _ in range(iters):
            u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
        return Z + u.unsqueeze(2) + v.unsqueeze(1)

    def log_optimal_transport(self, scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
        """ Perform Differentiable Optimal Transport in Log-space for stability"""
        b, m, n = scores.shape
        # one = scores.new_tensor(1)
        one = torch.ones((), device=scores.device)
        ms, ns = (m * one).to(scores), (n * one).to(scores)

        bins0 = alpha.expand(b, m, 1)
        bins1 = alpha.expand(b, 1, n)
        alpha = alpha.expand(b, 1, 1)

        couplings = torch.cat([torch.cat([scores, bins0], -1),
                               torch.cat([bins1, alpha], -1)], 1)

        norm = - (ms + ns).log()
        log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
        log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
        log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

        Z = self.log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
        Z = Z - norm  # multiply probabilities by M+N
        return Z.exp()

    def forward(self, dsc0, dsc1, tmp):

        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', dsc0, dsc1)
        scores = scores / self.descriptor_dim**.5

        # scores = torch.matmul(dsc0.transpose(1, 2).contiguous(), dsc1)

        scores = self.log_optimal_transport(
            scores, self.dustbin_score,
            iters=self.sinkhorn_iterations)

        return scores[:, :-1, :-1]
