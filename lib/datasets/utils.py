# Code adapted from Map-free benchmark: https://github.com/nianticlabs/map-free-reloc

import cv2
import numpy as np
import torch
from numpy.linalg import inv

def imread(path, augment_fn=None):
    cv_type = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), cv_type)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if augment_fn is not None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = augment_fn(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image  # (h, w, 3)


def get_resized_wh(w, h, resize=None):
    if resize is not None:  # resize the longer edge
        scale = resize / max(h, w)
        w_new, h_new = int(round(w * scale)), int(round(h * scale))
    else:
        w_new, h_new = w, h
    return w_new, h_new


def get_divisible_wh(w, h, df=None):
    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w, h])
    else:
        w_new, h_new = w, h
    return w_new, h_new


def pad_bottom_right(inp, pad_size, df, ret_mask=False, border=3):
    assert isinstance(pad_size, int) and pad_size >= max(
        inp.shape[-2:]), f"{pad_size} < {max(inp.shape[-2:])}"
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[:inp.shape[0], :inp.shape[1]] = True
    elif inp.ndim == 3:
        padded = np.zeros((pad_size, pad_size, inp.shape[2]), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1], :] = inp
        if ret_mask:

            mask = np.zeros((1, pad_size//df, pad_size//df))
            mask[:, :inp.shape[0]//df-border, :inp.shape[1]//df-border] = 1

    else:
        raise NotImplementedError()
    return padded, mask


def read_color_image(path, resize=(640, 480), augment_fn=None):
    """
    Args:
        resize (tuple): align image to depthmap, in (w, h).
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (3, h, w)
    """
    # read and resize image
    image = imread(path, None)
    image = cv2.resize(image, resize)

    # (h, w, 3) -> (3, h, w) and normalized
    image = torch.from_numpy(image).float().permute(2, 0, 1) / 255
    if augment_fn:
        image = augment_fn(image)
    return image


def read_depth_image(path):
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    depth = depth / 1000
    depth = torch.from_numpy(depth).float()  # (h, w)
    return depth

def correct_intrinsic_scale(K, scale_x, scale_y):
    '''Given an intrinsic matrix (3x3) and two scale factors, returns the new intrinsic matrix corresponding to
    the new coordinates x' = scale_x * x; y' = scale_y * y
    Source: https://dsp.stackexchange.com/questions/6055/how-does-resizing-an-image-affect-the-intrinsic-camera-matrix
    '''

    transform = torch.eye(3)
    transform[0, 0] = scale_x
    transform[0, 2] = scale_x / 2 - 0.5
    transform[1, 1] = scale_y
    transform[1, 2] = scale_y / 2 - 0.5
    Kprime = transform @ K

    return Kprime

def define_sampling_grid(im_size, feats_downsample=4, step=1):
    """
        Auxiliary function to generate the sampling grid from the feature map
        Args:
            im_size: original image size that goes into the network
            feats_downsample: rescaling factor that happens within the architecture due to downsampling steps
        Output:
            indexes_mat: dense grid sampling indexes, size: (im_size/feats_downsample, im_size/feats_downsample)
    """

    feats_size = int(im_size/feats_downsample)
    grid_size = int(im_size/feats_downsample/step)

    indexes = np.asarray(range(0, feats_size, step))[:grid_size]
    indexes_x = indexes.reshape((1, len(indexes), 1))
    indexes_y = indexes.reshape((len(indexes), 1, 1))

    indexes_x = np.tile(indexes_x, [len(indexes), 1, 1])
    indexes_y = np.tile(indexes_y, [1, len(indexes), 1])

    indexes_mat = np.concatenate([indexes_x, indexes_y], axis=-1)
    indexes_mat = indexes_mat.reshape((grid_size*grid_size, 2))

    return indexes_mat


