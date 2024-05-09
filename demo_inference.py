import torch
import argparse
from lib.models.builder import build_model
from lib.datasets.utils import correct_intrinsic_scale
from lib.models.MicKey.modules.utils.training_utils import colorize, generate_heat_map
from config.default import cfg
import numpy as np
from pathlib import Path
import cv2

def prepare_score_map(scs, img, temperature=0.5):

    score_map = generate_heat_map(scs, img, temperature)

    score_map = 255 * score_map.permute(1, 2, 0).numpy()

    return score_map

def colorize_depth(value, vmin=None, vmax=None, cmap='magma_r', invalid_val=-99, invalid_mask=None, background_color=(0, 0, 0, 255), gamma_corrected=False, value_transform=None):

    img = colorize(value, vmin, vmax, cmap, invalid_val, invalid_mask, background_color, gamma_corrected, value_transform)

    shape_im = img.shape
    img = np.asarray(img, np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    img = cv2.resize(img, (shape_im[1]*14, shape_im[0]*14), interpolation=cv2.INTER_LINEAR)

    return img

def read_color_image(path, resize):
    """
    Args:
        resize (tuple): align image to depthmap, in (w, h).
    Returns:
        image (torch.tensor): (3, h, w)
    """
    # read and resize image
    cv_type = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), cv_type)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if resize is not None:
        image = cv2.resize(image, resize)

    # (h, w, 3) -> (3, h, w) and normalized
    image = torch.from_numpy(image).float().permute(2, 0, 1) / 255

    return image.unsqueeze(0)

def read_intrinsics(path_intrinsics, resize):
    Ks = {}
    with Path(path_intrinsics).open('r') as f:
        for line in f.readlines():
            if '#' in line:
                continue

            line = line.strip().split(' ')
            img_name = line[0]
            fx, fy, cx, cy, W, H = map(float, line[1:])

            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            if resize is not None:
                K = correct_intrinsic_scale(K, resize[0] / W, resize[1] / H).numpy()
            Ks[img_name] = K
    return Ks

def run_demo_inference(args):

    # Select device
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    print('Preparing data...')

    # Prepare config file
    cfg.merge_from_file(args.config)

    # Prepare the model
    model = build_model(cfg, checkpoint=args.checkpoint)

    # Load demo images
    im0 = read_color_image(args.im_path_ref, args.resize).to(device)
    im1 = read_color_image(args.im_path_dst, args.resize).to(device)

    # Load intrinsics
    K = read_intrinsics(args.intrinsics, args.resize)

    # Prepare data for MicKey
    data = {}
    data['image0'] = im0
    data['image1'] = im1
    data['K_color0'] = torch.from_numpy(K['im0.jpg']).unsqueeze(0).to(device)
    data['K_color1'] = torch.from_numpy(K['im1.jpg']).unsqueeze(0).to(device)

    # Run inference
    print('Running MicKey relative pose estimation...')
    model(data)

    # Pose, inliers and score are stored in:
    # data['R'] = R
    # data['t'] = t
    # data['inliers'] = inliers
    # data['inliers_list'] = inliers_list

    print('Saving depth and score maps in image directory ...')
    depth0_map = colorize_depth(data['depth0_map'][0], invalid_mask=(data['depth0_map'][0] < 0.001).cpu()[0])
    depth1_map = colorize_depth(data['depth1_map'][0], invalid_mask=(data['depth1_map'][0] < 0.001).cpu()[0])
    score0_map = prepare_score_map(data['scr0'][0], data['image0'][0], temperature=0.5)
    score1_map = prepare_score_map(data['scr1'][0], data['image1'][0], temperature=0.5)

    ext_im0 = args.im_path_ref.split('.')[-1]
    ext_im1 = args.im_path_dst.split('.')[-1]

    cv2.imwrite(args.im_path_ref.replace(ext_im0, 'score.jpg'), score0_map)
    cv2.imwrite(args.im_path_dst.replace(ext_im1, 'score.jpg'), score1_map)

    cv2.imwrite(args.im_path_ref.replace(ext_im0, 'depth.jpg'), depth0_map)
    cv2.imwrite(args.im_path_dst.replace(ext_im1, 'depth.jpg'), depth1_map)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--im_path_ref', help='path to reference image', default='data/toy_example/im0.jpg')
    parser.add_argument('--im_path_dst', help='path to destination image', default='data/toy_example/im1.jpg')
    parser.add_argument('--intrinsics', help='path to intrinsics file', default='data/toy_example/intrinsics.txt')
    parser.add_argument('--resize', nargs=2, type=int, help='resize applied to the image and intrinsics (w, h)', default=None)
    parser.add_argument('--config', help='path to config file', default='weights/mickey_weights/config.yaml')
    parser.add_argument('--checkpoint', help='path to model checkpoint',
                        default='weights/mickey_weights/mickey.ckpt')
    args = parser.parse_args()

    run_demo_inference(args)

