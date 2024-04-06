import torch

def weighted_procrustes(A, B, w=None, use_weights=True, use_mask=False, eps=1e-16, check_rank=True):
    """
    X: torch tensor B x N x 3
    Y: torch tensor B x N x 3
    w: torch tensor B x N
    """
    # https://ieeexplore.ieee.org/document/88573
    # https://github.com/chrischoy/DeepGlobalRegistration/blob/master/core/registration.py#L160
    # Refer to Mapfree procrustes
    assert len(A) == len(B)
    if use_weights:
        W1 = torch.abs(w).sum(1, keepdim=True)
        w_norm = (w / (W1 + eps)).unsqueeze(-1)
        a_mean = (w_norm * A).sum(1, keepdim=True)
        b_mean = (w_norm * B).sum(1, keepdim=True)
        # print('Possible ERROR:')
        # print('check this: Sxy = (Y - muy).t().mm(w_norm * (X - mux)).cpu().double(). Repo DeepGlobalRegistration')

        A_c = A - a_mean
        B_c = B - b_mean

        if use_mask:
            # Covariance matrix
            H = A_c.transpose(1, 2) @ (w.unsqueeze(-1) * B_c)
        else:
            # Covariance matrix
            H = A_c.transpose(1, 2) @ (w_norm.unsqueeze(-1) * B_c)

    else:
        a_mean = A.mean(axis=1, keepdim=True)
        b_mean = B.mean(axis=1, keepdim=True)

        A_c = A - a_mean
        B_c = B - b_mean

        # Covariance matrix
        H = A_c.transpose(1, 2) @ B_c

    if check_rank:
        if (torch.linalg.matrix_rank(H) == 1).sum() > 0:
            return None, None, False

    U, S, V = torch.svd(H)
    # Fixes orientation such that Det(R) = + 1
    Z = torch.eye(3).unsqueeze(0).repeat(A.shape[0], 1, 1).to(A.device)
    Z[:, -1, -1] = torch.sign(torch.linalg.det(U @ V.transpose(1, 2)))
    # Rotation matrix
    R = V @ Z @ U.transpose(1, 2)
    # Translation vector
    t = b_mean - a_mean @ R.transpose(1, 2)

    return R, t, True