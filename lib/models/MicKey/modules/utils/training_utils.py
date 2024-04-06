import torch
import numpy as np
import cv2
import matplotlib
from os import mkdir, path

def backproject_3d(uv, depth, K):
    '''
    Backprojects 2d points given by uv coordinates into 3D using their depth values and intrinsic K
    :param uv: array [B, N, 2]
    :param depth: array [B, N, 1]
    :param K: array [B, 3, 3]
    :return: xyz: array [B, N, 3]
    '''

    B, num_corr, _ = uv.shape

    ones_vector = torch.ones((B, num_corr, 1)).to(uv.device)
    uv1 = torch.cat([uv, ones_vector], dim=-1)
    xyz = depth * (torch.linalg.inv(K) @ uv1.transpose(2, 1)).transpose(2, 1)

    return xyz

def project_2d(XYZ, K):
    '''
    Backprojects 3d points given by XYZ coordinates into 2D using their depth values and intrinsic K
    XYZ - Size: B, n, 3
    '''

    B, num_corr, _ = XYZ.shape

    xyz_cam = (K @ XYZ.transpose(2, 1)).transpose(2, 1)
    xy = xyz_cam / (xyz_cam[:, :, 2].view(B, num_corr, 1)+1e-16)

    return xy[:, :, :2]


def soft_inlier_counting(X0, xy1, R, t, K1, beta=100):
    B = X0.shape[0]
    X0_to_1 = (R @ X0.transpose(2, 1)).transpose(2, 1) + t
    kp0_to_1 = project_2d(X0_to_1, K1)
    scores = torch.sigmoid(beta - ((((kp0_to_1 - xy1) ** 2.).sum(-1) + 1e-6) ** 0.5)).sum(-1).view(B, 1)
    return scores

def soft_inlier_counting_dsac(X0, xy1, R, t, K1, th=10):
    B = X0.shape[0]
    beta = 5 / th
    X0_to_1 = (R @ X0.transpose(2, 1)).transpose(2, 1) + t
    kp0_to_1 = project_2d(X0_to_1, K1)
    dist = ((((kp0_to_1 - xy1) ** 2.).sum(-1) + 1e-6) ** 0.5)
    scores = torch.sigmoid(beta * (th - dist)).sum(-1).view(B, 1)
    return scores


def soft_inlier_counting_3d(X0, X1,  R, t, th=0.5):
    B = X0.shape[0]
    beta = 5 / th
    X0_to_1 = (R @ X0.transpose(2, 1)).transpose(2, 1) + t
    dist = ((((X0_to_1 - X1) ** 2.).sum(-1) + 1e-6) ** 0.5)
    scores = torch.sigmoid(beta * (th - dist)).sum(-1).view(B, 1)
    return scores

def weighted_soft_inlier_counting_3d(X0, X1, w, R, t, th=0.5):
    B = X0.shape[0]
    beta = 5 / th
    X0_to_1 = (R @ X0.transpose(2, 1)).transpose(2, 1) + t
    dist = ((((X0_to_1 - X1) ** 2.).sum(-1) + 1e-6) ** 0.5)
    scores = (w*torch.sigmoid(beta * (th - dist))).sum(-1).view(B, 1)
    return scores

def inlier_counting_3d(X0, X1,  R, t, th=0.5):
    X0_to_1 = (R @ X0.transpose(2, 1)).transpose(2, 1) + t
    dist = ((((X0_to_1 - X1) ** 2.).sum(-1) + 1e-6) ** 0.5)
    inliers = (th - dist) >= 0
    return inliers.float()


def inlier_counting(X0, xy1, R, t, K1, th_px=10):
    X0_to_1 = (R @ X0.transpose(2, 1)).transpose(2, 1) + t
    kp0_to_1 = project_2d(X0_to_1, K1)
    inliers = (th_px - ((((kp0_to_1 - xy1) ** 2.).sum(-1) + 1e-16) ** 0.5) > 0)
    return inliers

def check_out_border_kpt(kpt, shape_im, borders=50):

    non_border_kpts = (kpt[:, 0] > borders) * \
                      (kpt[:, 1] > borders) * \
                      (kpt[:, 0] < shape_im[1] - borders) * \
                      (kpt[:, 1] < shape_im[1] - borders)

    return non_border_kpts

def vis_inliers(inliers, batch, batch_i=0, num_max_matches=60, norm_color=True):

    # Check the matches:
    # matches_list = inliers[batch_i][:num_max_matches]
    matches_list = inliers[batch_i][np.random.permutation(len(inliers[batch_i]))][:num_max_matches]

    # Prepare the matching image:
    image0 = (255 * batch['image0'][batch_i].permute(1, 2, 0)).cpu().numpy()
    image1 = (255 * batch['image1'][batch_i].permute(1, 2, 0)).cpu().numpy()

    shape_im = image0.shape

    tmp_im = 255 * np.ones((max(shape_im[0], shape_im[0]), (shape_im[1] + shape_im[1]) + 50, 3))
    tmp_im[:shape_im[0], :shape_im[1], :] = image0
    tmp_im[:shape_im[0], shape_im[1] + 50:shape_im[1] + 50 + shape_im[1], :] = image1

    random_color = False

    for i_m in range(len(matches_list)):

        pt_ref = matches_list[i_m, :2].detach().cpu().numpy()
        pt_dst = matches_list[i_m, 2:4].detach().cpu().numpy()

        if random_color:
            color = list(np.random.uniform(low=0, high=255, size=(3,)))
            color[0], color[1], color[2] = int(color[0]), int(color[1]), int(color[2])
        else:
            if norm_color:
                scs = inliers[batch_i][:, 4].detach().cpu().numpy()
                sc = scs[i_m]/scs.max()
                color = [0, int(sc * 255), 0]
            else:
                color = [0, 225, 0]

        tmp_im = cv2.line(tmp_im, (int(pt_ref[0]), int(pt_ref[1])),
                          (shape_im[1] + 50 + int(pt_dst[0]), int(pt_dst[1])), color, 2)
        tmp_im = cv2.circle(tmp_im, (int(pt_ref[0]), int(pt_ref[1])), 5, color, 3)
        tmp_im = cv2.circle(tmp_im, (int(shape_im[1] + 50 + pt_dst[0]), int(pt_dst[1])), 5, color, 3)

    im_matches = torch.from_numpy(tmp_im).permute(2, 0, 1) / 255.
    return im_matches


def generate_heat_map(scs, img, temperature=0.5):

    sc_kp = scs[0, :]
    max_sc = max(sc_kp).detach().cpu().numpy()
    min_sc = min(sc_kp).detach().cpu().numpy()

    sc_map = np.ascontiguousarray((255 * img.permute(1, 2, 0)).cpu().numpy().astype(np.uint8))

    sc_map = cv2.cvtColor(sc_map, cv2.COLOR_BGR2GRAY)
    sc_map = np.tile(sc_map[:, :, np.newaxis], [1, 1, 3])

    shape_sc = [sc_map.shape[0]//14, sc_map.shape[1]//14]
    heat_map = sc_kp.reshape(shape_sc)
    heat_map = 255 * np.tanh(((heat_map.detach().cpu().numpy() - min_sc) / (max_sc - min_sc + 1e-16)) / temperature)
    heat_map = cv2.resize(heat_map, (sc_map.shape[1], sc_map.shape[0]))[:, :, np.newaxis]
    heat_map = np.concatenate([np.zeros_like(heat_map), heat_map, np.zeros_like(heat_map)], axis=-1)

    sc_map = cv2.addWeighted(sc_map, 1., np.asarray(heat_map, np.uint8), 0.6, 0)

    return torch.from_numpy(sc_map).permute(2, 0, 1) / 255.


def generate_kp_im(scs, img, kps, temperature=0.3):
    # color = np.asarray([0, 255, 0], np.float)
    color = np.asarray([0, 255, 0], float)
    sc_kp = scs[0, :]
    max_sc = max(sc_kp).detach().cpu().numpy()
    min_sc = min(sc_kp).detach().cpu().numpy()

    sc_map = np.ascontiguousarray((255 * img.permute(1, 2, 0)).cpu().numpy().astype(np.uint8))

    for i_m in range(len(sc_kp)):
        pt_ref = kps[:, i_m].detach().cpu().numpy()
        sc_kp = scs[0, i_m].detach().cpu().numpy()

        # Normalise score for better visualisation
        sc_matching = (sc_kp - min_sc) / (max_sc - min_sc + 1e-16)
        color_tmp = color * np.tanh(sc_matching / temperature)
        sc_map = cv2.circle(sc_map, (int(pt_ref[0]), int(pt_ref[1])), 2, color_tmp, 2)

    return torch.from_numpy(sc_map).permute(2, 0, 1) / 255.

def colorize(value, vmin=None, vmax=None, cmap='magma_r', invalid_val=-99, invalid_mask=None, background_color=(0, 0, 0, 255), gamma_corrected=False, value_transform=None):
    """Converts a depth map to a color image: https://github.com/isl-org/ZoeDepth/blob/main/zoedepth/utils/misc.py

    Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r', gray_r
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    else:
        if (invalid_mask==False).sum()==0:
            invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask==1],2) if vmin is None else vmin
    vmax = np.percentile(value[mask==1],85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
    value = cmapper(value, bytes=True)  # (nxmx4)

    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img

def log_image_matches(matcher, batch, train_depth=False, batch_i=0, num_vis_matches=30, sc_temp=0.3, norm_color=True):

    # Plot to kps
    sc_map0 = generate_kp_im(batch['scr0'][batch_i], batch['image0'][batch_i], batch['kps0'][batch_i], temperature=sc_temp)
    sc_map1 = generate_kp_im(batch['scr1'][batch_i], batch['image1'][batch_i], batch['kps1'][batch_i], temperature=sc_temp)

    if train_depth:
        border_sz = 6
        _, _, dh, dw = batch['depth0_map'].size()

        depth0_map = batch['depth0_map'][batch_i].detach()
        depth1_map = batch['depth1_map'][batch_i].detach()

        depth_map0 = torch.from_numpy(colorize(depth0_map, invalid_mask=(depth0_map < 0.001).cpu()[0])/255.).permute(2, 0, 1)[:3]
        depth_map1 = torch.from_numpy(colorize(depth1_map, invalid_mask=(depth1_map < 0.001).cpu()[0])/255.).permute(2, 0, 1)[:3]

        depth0_map_n_borders = batch['depth0_map'][batch_i][:, border_sz:dh - border_sz, border_sz:dw - border_sz].detach()
        depth1_map_n_borders = batch['depth1_map'][batch_i][:, border_sz:dh - border_sz, border_sz:dw - border_sz].detach()

        depth0_map_n_borders = torch.from_numpy(colorize(depth0_map_n_borders)).permute(2, 0, 1)[:3]
        depth1_map_n_borders = torch.from_numpy(colorize(depth1_map_n_borders)).permute(2, 0, 1)[:3]

        depth_map0 = [depth_map0, depth0_map_n_borders]
        depth_map1 = [depth_map1, depth1_map_n_borders]
    else:
        depth_map0 = None
        depth_map1 = None

    # Prepare the matching image:
    image0 = (255 * batch['image0'][batch_i].permute(1, 2, 0)).detach().cpu().numpy()
    image1 = (255 * batch['image1'][batch_i].permute(1, 2, 0)).detach().cpu().numpy()

    shape_im = image0.shape

    tmp_im = 255 * np.ones((max(shape_im[0], shape_im[0]), (shape_im[1] + shape_im[1]) + 50, 3))
    tmp_im[:shape_im[0], :shape_im[1], :] = image0
    tmp_im[:shape_im[0], shape_im[1] + 50:shape_im[1] + 50 + shape_im[1], :] = image1

    # Check the matches:
    matches_list = matcher.get_matches_list(batch['final_scores'][batch_i].detach().unsqueeze(0))

    if len(matches_list) == 0:
        im_matches = torch.from_numpy(tmp_im).permute(2, 0, 1) / 255.
        return im_matches, sc_map0, sc_map1, depth_map0, depth_map1

    corr0 = batch['kps0'][batch_i, :2, matches_list[:, 0]].T.detach()
    corr1 = batch['kps1'][batch_i, :2, matches_list[:, 1]].T.detach()

    non_border_matches0 = check_out_border_kpt(corr0, shape_im, borders=50)
    non_border_matches1 = check_out_border_kpt(corr1, shape_im, borders=50)

    non_border_matches = non_border_matches0 * non_border_matches1
    matches_list = matches_list[non_border_matches]

    if len(matches_list) == 0:
        im_matches = torch.from_numpy(tmp_im).permute(2, 0, 1) / 255.
        return im_matches, sc_map0, sc_map1, depth_map0, depth_map1

    # Sort by matching score
    color = np.asarray([0, 255, 0], float)
    sc_matching = batch['scores'][batch_i, matches_list[:, 0], matches_list[:, 1]].detach()
    matches_list = matches_list[torch.argsort(sc_matching, descending=True)]
    max_sc = max(sc_matching).detach().cpu().numpy()
    min_sc = min(sc_matching).detach().cpu().numpy()

    for i_m in range(min(num_vis_matches, len(matches_list))):
        match = matches_list[i_m]
        pt_ref = batch['kps0'][batch_i, :, match[0]].detach().cpu().numpy()
        pt_dst = batch['kps1'][batch_i, :, match[1]].detach().cpu().numpy()
        sc_matching = batch['scores'][batch_i, match[0], match[1]].detach().cpu().numpy()

        # Normalise score for better visualisation
        sc_matching = (sc_matching - min_sc) / (max_sc - min_sc + 1e-16)
        if norm_color:
            color_tmp = color * np.tanh(sc_matching/0.3)
        else:
            color_tmp = color

        tmp_im = cv2.line(tmp_im, (int(pt_ref[0]), int(pt_ref[1])), (shape_im[1] + 50 + int(pt_dst[0]), int(pt_dst[1])), color_tmp, 2)

        tmp_im = cv2.circle(tmp_im, (int(pt_ref[0]), int(pt_ref[1])), 2, color, 2)
        tmp_im = cv2.circle(tmp_im, (int(shape_im[1] + 50 + pt_dst[0]), int(pt_dst[1])), 2, color, 2)

    im_matches = torch.from_numpy(tmp_im).permute(2, 0, 1) / 255.
    return im_matches, sc_map0, sc_map1, depth_map0, depth_map1


def debug_reward_matches_log(data, gradients, batch_i = 0, num_vis_pts = 30):

    image0 = (255 * data['image0'][batch_i].permute(1, 2, 0)).cpu().numpy()
    image1 = (255 * data['image1'][batch_i].permute(1, 2, 0)).cpu().numpy()

    shape_im = image0.shape

    tmp_im = 255 * np.ones((max(shape_im[0], shape_im[0]), (shape_im[1] + shape_im[1]) + 50, 3))
    tmp_im[:shape_im[0], :shape_im[1], :] = image0
    tmp_im[:shape_im[0], shape_im[1] + 50:shape_im[1] + 50 + shape_im[1], :] = image1

    kps0 = data['kps0']
    kps1 = data['kps1']

    gradients_dsc = gradients[0]

    B, num_kpts, _ = gradients_dsc.shape
    gradients_dsc = gradients_dsc.reshape(B, num_kpts * num_kpts)

    active_grads = torch.where(gradients_dsc[batch_i] != 0.)[0]
    sampled_idx_kp0 = torch.div(active_grads, num_kpts, rounding_mode='trunc')
    sampled_idx_kp1 = (active_grads % num_kpts)

    cor0 = kps0[batch_i, :2, sampled_idx_kp0].T.detach().cpu().numpy()
    cor1 = kps1[batch_i, :2, sampled_idx_kp1].T.detach().cpu().numpy()

    gradients_i = gradients_dsc[batch_i][active_grads].detach().cpu().numpy()

    # High gradients push down matching values.
    grad_idx = gradients_i - gradients_i.min()
    grad_idx = 1 - grad_idx/grad_idx.max()

    idx_random = np.arange(len(cor0))[np.argsort(grad_idx)][:num_vis_pts//2]
    for i_m in range(len(idx_random)):
        pt_ref = [int(cor0[idx_random[i_m]][0]), int(cor0[idx_random[i_m]][1])]
        pt_dst = [int(cor1[idx_random[i_m]][0]), int(cor1[idx_random[i_m]][1])]
        if grad_idx[idx_random[i_m]] < 0.5:
            color = [int((1-grad_idx[idx_random[i_m]]) * 255), 0, 0]
        else:
            color = [0, int(grad_idx[idx_random[i_m]] * 255), 0]
        tmp_im = cv2.line(tmp_im, (int(pt_ref[0]), int(pt_ref[1])),
                          (shape_im[1] + 50 + int(pt_dst[0]), int(pt_dst[1])), color, 1)
        tmp_im = cv2.circle(tmp_im, (int(pt_ref[0]), int(pt_ref[1])), 8, color, 2)
        tmp_im = cv2.circle(tmp_im, (int(shape_im[1] + 50 + pt_dst[0]), int(pt_dst[1])), 8, color, 2)

    idx_random = np.arange(len(cor0))[np.argsort(grad_idx)[::-1]][:num_vis_pts//2]
    for i_m in range(len(idx_random)):
        pt_ref = [int(cor0[idx_random[i_m]][0]), int(cor0[idx_random[i_m]][1])]
        pt_dst = [int(cor1[idx_random[i_m]][0]), int(cor1[idx_random[i_m]][1])]
        if grad_idx[idx_random[i_m]] < 0.5:
            color = [int((1-grad_idx[idx_random[i_m]]) * 255), 0, 0]
        else:
            color = [0, int(grad_idx[idx_random[i_m]] * 255), 0]
        tmp_im = cv2.line(tmp_im, (int(pt_ref[0]), int(pt_ref[1])),
                          (shape_im[1] + 50 + int(pt_dst[0]), int(pt_dst[1])), color, 1)
        tmp_im = cv2.circle(tmp_im, (int(pt_ref[0]), int(pt_ref[1])), 8, color, 2)
        tmp_im = cv2.circle(tmp_im, (int(shape_im[1] + 50 + pt_dst[0]), int(pt_dst[1])), 8, color, 2)

    im_rewards = torch.from_numpy(tmp_im).permute(2, 0, 1) / 255.
    rew_kp0, rew_kp1 = None, None

    return im_rewards, rew_kp0, rew_kp1

def create_exp_name(exp_name, cfg):

    exp_name += ('_Loss_' + cfg.LOSS_CLASS.LOSS_FUNCTION)
    exp_name += '_SoftClipping' if cfg.LOSS_CLASS.SOFT_CLIPPING else ''

    if cfg.LOSS_CLASS.CURRICULUM_LEARNING.TRAIN_CURRICULUM:
        exp_name += '_Curriculum'
    elif cfg.LOSS_CLASS.CURRICULUM_LEARNING.TRAIN_WITH_TOPK:
        exp_name += ('_TrainTop' + str(cfg.LOSS_CLASS.CURRICULUM_LEARNING.TOPK))

    exp_name += '_NullHypothesis' if cfg.LOSS_CLASS.NULL_HYPOTHESIS.ADD_NULL_HYPOTHESIS else ''

    exp_name += ('_DepthSigmoid_' + str(int(cfg.MICKEY.KP_HEADS.MAX_DEPTH))) if cfg.MICKEY.KP_HEADS.USE_DEPTHSIGMOID else ''

    exp_name += ('_' + cfg.FEATURE_MATCHER.TYPE)

    exp_name += '_Debug' if cfg.DEBUG else ''

    return exp_name

def create_result_dir(result_path):
    '''
        It creates the directory where features will be stored
    '''
    directories = result_path.split('/')
    tmp = ''
    for idx, dir in enumerate(directories):
        tmp += (dir + '/')
        if idx == len(directories)-1:
            continue
        if not path.isdir(tmp):
            mkdir(tmp)
