import torch
import math
from lib.utils.metrics import vcre_loss

def compute_angular_error(R, t, Rgt_i, tgt_i):
    loss_rot, rot_err = rot_angle_loss(R, Rgt_i)
    loss_trans, t_err = trans_ang_loss(t, tgt_i)

    max_loss, _ = torch.max(torch.cat((loss_rot, loss_trans), dim=-1), dim=-1)
    return max_loss, loss_rot, loss_trans

def compute_angular_error_weighted(R, t, Rgt_i, tgt_i, weights_t):
    loss_rot, rot_err = rot_angle_loss(R, Rgt_i)
    loss_trans, t_err = trans_ang_loss(t, tgt_i)

    max_loss, _ = torch.max(torch.cat((loss_rot, loss_trans * weights_t), dim=-1), dim=-1)
    return max_loss, loss_rot, loss_trans

def ess_sq_euclidean_error(E, Egt):

    B = E.shape[0]
    E_norm = E/E[:, 2, 2].view(B, 1, 1)
    Egt_norm = Egt/Egt[:, 2, 2].view(B, 1, 1)
    return torch.pow(E_norm-Egt_norm, 2).view(B, -1).sum(1)

def compute_pose_loss(R, t, Rgt_i, tgt_i, K=None, soft_clipping=True):
    loss_rot, rot_err = rot_angle_loss(R, Rgt_i)
    loss_trans = trans_l1_loss(t, tgt_i)

    if soft_clipping:
        loss_trans_soft = torch.tanh(loss_trans/0.9) # xm ~ ?
        loss_rot_soft = torch.tanh(loss_rot/0.9) # xrads=xdeg ~ ?

        loss = loss_rot_soft + loss_trans_soft
    else:
        loss = loss_rot + loss_trans

    return loss, loss_rot, loss_trans

def compute_vcre_loss(R, t, Rgt_i, tgt_i, K=None, soft_clipping=True):

    B = R.shape[0]
    Tgt = torch.zeros((B, 4, 4)).float().to(R.device)
    Tgt[:, :3, :3] = Rgt_i
    Tgt[:, :3, 3:] = tgt_i.transpose(2, 1)

    loss = vcre_loss(R, t, Tgt, K)
    if soft_clipping:
        loss = torch.tanh(loss/80)

    loss_rot, rot_err = rot_angle_loss(R, Rgt_i)
    loss_trans = trans_l1_loss(t, tgt_i)

    return loss, loss_rot, loss_trans

def trans_ang_loss(t, tgt):
    """Computes L1 loss for translation vector ANGULAR error
    Input:
    t - estimated translation vector [B, 1, 3]
    tgt - ground-truth translation vector [B, 1, 3]
    Output: translation_loss
    """

    scale_t = torch.linalg.norm(t, dim=-1)
    scale_tgt = torch.linalg.norm(tgt, dim=-1)

    cosine = (t @ tgt.transpose(1, 2)).squeeze(-1) / (scale_t * scale_tgt + 1e-6)
    cosine = torch.clip(cosine, -0.99999, 0.99999)  # handle numerical errors and NaNs
    t_ang_err = torch.acos(cosine)
    t_ang_err = torch.minimum(t_ang_err, math.pi - t_ang_err)
    return torch.abs(t_ang_err - torch.zeros_like(t_ang_err)), t_ang_err

def trans_l1_loss(t, tgt):
    """Computes L1 loss for translation vector
    Input:
    t - estimated translation vector [B, 1, 3]
    tgt - ground-truth translation vector [B, 1, 3]
    Output: translation_loss
    """

    return torch.abs(t - tgt).sum(-1)

def rot_angle_loss(R, Rgt):
    """
    Computes rotation loss using L1 error of residual rotation angle [radians]
    Input:
    R - estimated rotation matrix [B, 3, 3]
    Rgt - groundtruth rotation matrix [B, 3, 3]
    Output:  rotation_loss
    """

    residual = R.transpose(1, 2) @ Rgt
    trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
    cosine = (trace - 1) / 2
    cosine = torch.clip(cosine, -0.99999, 0.99999)  # handle numerical errors and NaNs
    R_err = torch.acos(cosine)
    loss = torch.abs(R_err - torch.zeros_like(R_err)).unsqueeze(-1)
    return loss, R_err


def to_homogeneous_torch_batched(u_xys: torch.Tensor):
    batch_size, _, num_pts = u_xys.shape
    ones = torch.ones((batch_size, 1, num_pts)).float().to(u_xys.device)
    u_xyhs = torch.concat([u_xys, ones], dim=1)
    return u_xyhs


