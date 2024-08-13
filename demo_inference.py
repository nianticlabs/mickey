import torch
import argparse
from lib.models.builder import build_model
from lib.datasets.utils import correct_intrinsic_scale
from lib.utils.visualization import prepare_score_map, colorize_depth, get_render, create_point_cloud_from_inliers
from config.default import cfg
import numpy as np
from pathlib import Path
import cv2


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

def generate_3d_vis(data, n_matches, root_dir, batch_id,
                    use_3d_color_coded=True, color_src_frame=[223, 71, 28], color_dst_frame=[83, 154, 218],
                    add_dst_lines=True, add_ref_lines=True, add_ref_pts=True, add_points=True,
                    size_box=0.03, size_2d=0.015, cam_size=1., max_conf_th=0.8, add_confidence=True,
                    angle_y=0, angle_x=-25, cam_offset_x=0.1, cam_offset_y=0.0, cam_offset_z=-2):

    print('Generating 3D visualization image...')

    # Generate point cloud from inlier matches
    point_cloud = create_point_cloud_from_inliers(data, batch_id, use_3d_color_coded)

    # Prepare the data:
    data_np = {}
    data_np['K_color0'] = data['K_color0'].detach().cpu().numpy()
    data_np['K_color1'] = data['K_color1'].detach().cpu().numpy()
    data_np['image0'] = 255 * data['image0'].permute(0, 2, 3, 1).detach().cpu().numpy()
    data_np['image1'] = 255 * data['image1'].permute(0, 2, 3, 1).detach().cpu().numpy()

    R = data['R'][batch_id][np.newaxis].detach().cpu().numpy()
    t = data['t'][batch_id].detach().cpu().numpy().reshape(-1)
    P = np.eye(4)
    P[:3, :3] = R
    P[:3, 3] = t

    # Render the image with camera and 3D points
    frame = get_render(P, data, batch_id, point_cloud, color_src_frame, color_dst_frame,
                       angle_y, angle_x, cam_offset_x, cam_offset_y, cam_offset_z, cam_size, size_box, size_2d,
                       add_ref_lines, add_dst_lines, add_ref_pts, add_points, n_matches, max_conf_th, add_confidence)

    cv2.imwrite(root_dir + '/3d_vis.png', cv2.cvtColor(np.uint8(frame), cv2.COLOR_BGR2RGB))

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
    batch_id = 0
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
    depth0_map = colorize_depth(data['depth0_map'][batch_id], invalid_mask=(data['depth0_map'][batch_id] < 0.001).cpu()[0])
    depth1_map = colorize_depth(data['depth1_map'][batch_id], invalid_mask=(data['depth1_map'][batch_id] < 0.001).cpu()[0])
    score0_map = prepare_score_map(data['scr0'][batch_id], data['image0'][batch_id], temperature=0.5)
    score1_map = prepare_score_map(data['scr1'][batch_id], data['image1'][batch_id], temperature=0.5)

    ext_im0 = args.im_path_ref.split('.')[-1]
    ext_im1 = args.im_path_dst.split('.')[-1]

    cv2.imwrite(args.im_path_ref.replace(ext_im0, 'score.jpg'), score0_map)
    cv2.imwrite(args.im_path_dst.replace(ext_im1, 'score.jpg'), score1_map)

    cv2.imwrite(args.im_path_ref.replace(ext_im0, 'depth.jpg'), depth0_map)
    cv2.imwrite(args.im_path_dst.replace(ext_im1, 'depth.jpg'), depth1_map)

    if args.generate_3D_vis:
        # We use the maximum possible number of inliers to draw the confidence
        n_matches = model.e2e_Procrustes.num_samples_matches
        dir_name = '/'.join(args.im_path_ref.split('/')[:-1])
        generate_3d_vis(data, n_matches, dir_name, batch_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--im_path_ref', help='path to reference image', default='data/toy_example/im0.jpg')
    parser.add_argument('--im_path_dst', help='path to destination image', default='data/toy_example/im1.jpg')
    parser.add_argument('--intrinsics', help='path to intrinsics file', default='data/toy_example/intrinsics.txt')
    parser.add_argument('--resize', nargs=2, type=int, help='resize applied to the image and intrinsics (w, h)', default=None)
    parser.add_argument('--config', help='path to config file', default='weights/mickey_weights/config.yaml')
    parser.add_argument('--checkpoint', help='path to model checkpoint',
                        default='weights/mickey_weights/mickey.ckpt')
    parser.add_argument('--generate_3D_vis', help='Set to True to generate a 3D visualisation of the output poses',
                        default=False)
    args = parser.parse_args()

    run_demo_inference(args)

