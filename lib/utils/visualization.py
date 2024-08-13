# Code to create MicKey webpage visualizations.
# Code to visualise the 3D assets is inspired on the following repositories - visit them for more visualizations:
# [CVPR23]: Accelerated Coordinate Encoding (ACE): Learning to Relocalize in Minutes using RGB and Poses (https://github.com/nianticlabs/ace)
# [ECCV24]: Scene Coordinate Reconstruction (ACE0): Posing of Image Collections via Incremental Learning of a Relocalizer (https://github.com/nianticlabs/acezero)
# [CVPR24]: DUSt3R: Geometric 3D Vision Made Easy (https://github.com/naver/dust3r)

from lib.models.MicKey.modules.utils.training_utils import colorize, generate_heat_map
from lib.models.MicKey.modules.utils.training_utils import backproject_3d
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import pyrender
import trimesh
import PIL
import os

# Used to render the images when the computer does not have a screen, eg, server
os.environ['PYOPENGL_PLATFORM'] = 'egl'

OPENGL = np.array([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])

def prepare_score_map(scs, img, temperature=0.5):
    """
        Generate the score map for visualization given the keypoint scores.
        Change the temperature to modify the smoothness of the score map.
    """
    score_map = generate_heat_map(scs, img, temperature)

    score_map = 255 * score_map.permute(1, 2, 0).numpy()

    return score_map

def colorize_depth(value, vmin=None, vmax=None, cmap='magma_r', invalid_val=-99, invalid_mask=None, background_color=(0, 0, 0, 255), gamma_corrected=False, value_transform=None):
    """
        Generate a normalized depth map.
        The depth map image is resized to the input resolution for visualization/inspection purposes.
    """

    img = colorize(value, vmin, vmax, cmap, invalid_val, invalid_mask, background_color, gamma_corrected, value_transform)

    shape_im = img.shape
    img = np.asarray(img, np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    img = cv2.resize(img, (shape_im[1]*14, shape_im[0]*14), interpolation=cv2.INTER_LINEAR)

    return img

def create_point_cloud_from_inliers(data, i_batch, use_3d_color_coded):
    """
        Generate a point cloud with corresponding point colors. Point cloud contains the 2D correspondence inliers,
        such that when visualizing it, it shows which points produced the final pose.
        The points are color-coded based on their position, if desired, you can set use_3d_color_coded=False to use the
        color of the original pixel.
    """

    inliers = data['inliers_list'][i_batch]
    xy0 = inliers[:, :2].unsqueeze(0)
    z0 = inliers[:, 5].unsqueeze(-1).unsqueeze(0)

    X = backproject_3d(xy0, z0, data['K_color0'][i_batch].unsqueeze(0))

    if use_3d_color_coded:
        blue = X[0, :, 0] - X[0, :, 0].min()
        blue = (255 * blue / blue.max()).cpu().numpy().astype(np.uint8)
        red = X[0, :, 1] - X[0, :, 1].min()
        red = (255 * red / red.max()).cpu().numpy().astype(np.uint8)
        green = X[0, :, 2] - X[0, :, 2].min()
        green = (255 * green / green.max()).cpu().numpy().astype(np.uint8)
        colors_pts = np.asarray([blue, red, green]).T
    else:
        x1_norm = 2. * xy0[:, :, 0] / (data['image0'].shape[3] - 1) - 1.0
        y1_norm = 2. * xy0[:, :, 1] / (data['image0'].shape[2] - 1) - 1.0
        grid_sample = torch.stack((x1_norm.unsqueeze(-1), y1_norm.unsqueeze(-1)), 3)
        colot_tmp = torch.nn.functional.grid_sample(data['image0'][i_batch].unsqueeze(0), grid_sample, mode='bilinear', align_corners=True)
        colors_pts = 255 * colot_tmp[0, :, :, 0].T.cpu().numpy()

    return [X[0].cpu().numpy(), colors_pts]

def transform_3Dpoints(G, pts):
    """
        Auxiliary function that applies a transformation G to a set of 3D point locations.
    """
    pts_shape = pts.shape
    pts_projected = pts @ G.T[:-1, :] + G.T[-1:, :]
    pts_projected = pts_projected[..., :pts_shape[-1]].reshape(pts_shape)
    return pts_projected

def generate_camera(scene, pose_c2w, color, im, f, alpha=255, cam_size=0.3, inv_edge_width=0.93):
    """
        pose_c2w: position of the camera within the scene
        im: image to display in the camera
        f: focal length
        color: color of the image frame
        cam_size controls the size of the camera mesh
    """
    # Image shape
    H, W, _ = im.shape

    # Define camera cone
    im_size = 0.99 * cam_size
    h_im = f * im_size / H
    w_im = im_size * 0.5 ** 0.5
    cam = trimesh.creation.cone(w_im, h_im, sections=4)

    # Add the image to the cone image
    rotz = np.eye(4)
    rotz[:3, :3] = Rotation.from_euler('z', np.deg2rad(45)).as_matrix()
    rotz[2, 3] = -h_im  # set the tip of the cone = optical center
    aspect_ratio = np.eye(4)
    aspect_ratio[0, 0] = W / H
    g = pose_c2w @ OPENGL @ aspect_ratio @ rotz
    vertices_im = transform_3Dpoints(g, cam.vertices[[4, 5, 1, 3]])
    vertices = transform_3Dpoints(g, cam.vertices[[4, 5, 1, 3]])
    faces = np.array([[0, 1, 2], [0, 2, 3], [2, 1, 0], [3, 2, 0]])
    img = trimesh.Trimesh(vertices=vertices, faces=faces)
    uv_coords = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])

    # Add the image to the texture of the mesh
    pil_im = PIL.Image.fromarray(im)
    img.visual = trimesh.visual.TextureVisuals(uv_coords, image=pil_im)

    # Add a new camera slightly bigger to visualize better the edges
    h_cam = f * cam_size / H
    w_cam = cam_size * 0.5 ** 0.5
    cam_f = trimesh.creation.cone(w_cam, h_cam, sections=4)
    faces = []
    for face in cam_f.faces:
        if 0 in face:
            continue
        a, b, c = face
        a2, b2, c2 = face + len(cam_f.vertices)
        a3, b3, c3 = face + 2 * len(cam_f.vertices)

        # add 3 pseudo-edges
        faces.append((a, b, b2))
        faces.append((a, a2, c))
        faces.append((c2, b, c))

        faces.append((a, b, b3))
        faces.append((a, a3, c))
        faces.append((c3, b, c))

    faces += [(c, b, a) for a, b, c in faces]

    # Define vertices
    vertices = np.r_[cam_f.vertices, inv_edge_width * cam_f.vertices, cam_f.vertices]
    vertices = transform_3Dpoints(g, vertices)
    cam = trimesh.Trimesh(vertices=vertices, faces=faces)
    cam.visual.face_colors[:, :4] = color + [np.uint8(alpha)]

    # Add assets to scene
    scene.add_geometry(img)
    scene.add_geometry(cam)

    return vertices_im


def create_scene_and_cams(pose_c2w, data, batch_id, color_src_frame=[255, 0, 0], color_dst_frame=[0, 255, 0],
                          cam_size=0.05, alpha_value=255):
    """
        Define the scene and the cameras. The function also draws the corresponding images inside the cameras.
    """

    # Create Scene
    scene = trimesh.Scene()

    # Define focal lengths
    focal0 = ((data['K_color0'][batch_id, 0, 0] + data['K_color0'][batch_id, 1, 1]) / 2).item()
    focal1 = ((data['K_color1'][batch_id, 0, 0] + data['K_color1'][batch_id, 1, 1]) / 2).item()

    # Torch images to numpy
    img0 = np.uint8(255 * data['image0'].permute(0, 2, 3, 1)[batch_id].detach().cpu().numpy())
    img1 = 255 * data['image1'].permute(0, 2, 3, 1)[batch_id].detach().cpu().numpy()

    # Option to use the alpha value as the confidence to have transparent frames when low confidence
    img1_alpha = alpha_value * np.ones((img0.shape[0], img0.shape[1], 1))
    img1 = np.uint8(np.concatenate([img1, img1_alpha], axis=2))

    # Generate reference camera
    im_plane0 = generate_camera(scene, np.eye(4), color_src_frame,
                  img0, focal0, cam_size=cam_size)

    # Generate destination camera
    im_plane1 = generate_camera(scene, pose_c2w, color_dst_frame,
                  img1, focal1, alpha_value, cam_size=cam_size)

    return scene, im_plane0, im_plane1


def convert_face_colors_to_vertex_colors(mesh):
    if hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
        # Set the vertex colors
        mesh.visual.vertex_colors = np.tile(np.asarray(mesh.visual.face_colors[0]), [mesh.vertices.shape[0], 1])

def get_render(pose, data, batch_id, point_cloud, color_src_frame=[255, 0, 0], color_dst_frame=[0, 255, 0],
               angle_y=0, angle_x=-15, cam_offset_x=0.05, cam_offset_y=-0.02, cam_offset_z=-2,
               cam_size=1., size_box=0.03, size_2d=0.015, add_ref_lines=True, add_dst_lines=True,
               add_ref_pts=True, add_3d_points=True, total_matches=2000, max_conf_th=0.8, add_confidence=True):

    """
    Generate a scene with images, 3D points and (optionally) pose confidence.

    pose: Relative pose between the two input frames
    data: Dictionary containing intrinsics and images
    batch_id: batch id of the pair of images to visualize
    angle_y: angle in y-axis of the rendered camera
    angle_x: angle in x-axis of the rendered camera
    cam_offset_x: position in x-axis of the rendered camera wrt origin
    cam_offset_y: position in y-axis of the rendered camera wrt origin
    cam_offset_z: position in z-axis of the rendered camera wrt origin
    color_src_frame: color for reference frame
    color_dst_frame: color for destination frame
    cam_size defines the size of the camera frames
    size_box defines the size of the 3D points
    size_2d defines the size of the 2D points in the image plane
    add_ref_lines defines whether to visualize lines from 3D points to 2D points in the reference frame
    add_dst_lines defines whether to visualize lines from 3D points to 2D points in the destination frame
    add_ref_pts defines whether to visualize the 2D points in the reference frame
    add_3d_points defines whether to visualize the 3D points
    total_matches is the maximum number of possible inliers between the two frames
    max_conf_th controls the confidence visualization. Set to 1 to visualize the real confidence of the network
    add_confidence indicates whether confidence should be rendered or not

    Returns the render image with the images, 3D points, 2D points and (optionally) pose confidence.
    """

    # Define the position of the camera - this might need to change depending on your image pair
    center = point_cloud[0].mean(0)

    # create cameras
    pose_c2w = np.linalg.inv(pose)
    scene_cams, im_plane0, im_plane1 = create_scene_and_cams(pose_c2w, data, batch_id,
                                                             color_src_frame, color_dst_frame, cam_size)

    # Add scene rotation
    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene_cams.apply_transform(np.linalg.inv(OPENGL @ rot))

    # Create a scene for rendering
    scene = pyrender.Scene(ambient_light=[1., 1., 1.])

    # Iterate over the meshes in the scene_cams
    for m in scene_cams.geometry.values():
        # Set the vertex colors
        if hasattr(m.visual, 'face_colors') and m.visual.face_colors is not None:
            m.visual.vertex_colors = np.tile(np.asarray(m.visual.face_colors[0]), [m.vertices.shape[0], 1])

        # Add the mesh to the pyrender scene
        scene.add(pyrender.Mesh.from_trimesh(m))

    # Add point cloud
    pts3d = point_cloud[0]
    pts_colors = point_cloud[1]
    extents = [size_box, size_box, size_box]
    if add_3d_points:
        for idx, pt in enumerate(pts3d):
            box = trimesh.creation.box(extents=extents, transform=trimesh.transformations.translation_matrix(pt))
            box.visual.vertex_colors = pts_colors[idx].astype(np.uint8)
            scene.add(pyrender.Mesh.from_trimesh(box))

    # Add reference lines and/or 2D points
    if add_ref_lines or add_ref_pts:

        extents = [size_2d, size_2d, size_2d]
        origin = np.array([0, 0, 0]).reshape(3,)

        # Calculate the normal to the plane
        plane_normal = np.cross(im_plane0[1] - im_plane0[0], im_plane0[2] - im_plane0[0])
        dot_product = np.dot(plane_normal, (im_plane0[0] - origin))

        for idx in range(len(pts3d)):

            line_direction = pts3d[idx] - origin
            t = dot_product / np.dot(plane_normal, line_direction)
            intersection_point = origin + t * line_direction

            if add_ref_lines:
                points = [intersection_point, pts3d[idx]]
                line = trimesh.creation.cylinder(radius=0.0015, segment=points)
                line.visual.vertex_colors = pts_colors[idx].astype(np.uint8)
                scene.add(pyrender.Mesh.from_trimesh(line))

            if add_ref_pts:
                box = trimesh.creation.box(extents=extents,
                                           transform=trimesh.transformations.translation_matrix(intersection_point))
                box.visual.vertex_colors = pts_colors[idx].astype(np.uint8)
                scene.add(pyrender.Mesh.from_trimesh(box))

    # Add destination lines
    if add_dst_lines:

        extents = [size_2d, size_2d, size_2d]
        origin_dst = pose_c2w[:3, 3]

        # Calculate the normal to the plane
        plane_normal = np.cross(im_plane1[1] - im_plane1[0], im_plane1[2] - im_plane1[0])
        dot_prod = np.dot(plane_normal, (im_plane1[0] - origin_dst))

        for idx in range(len(pts3d)):

            line_direction = pts3d[idx] - origin_dst
            t = dot_prod / np.dot(plane_normal, line_direction)
            intersection_point = origin_dst + t * line_direction

            points = [intersection_point, pts3d[idx]]
            line = trimesh.creation.cylinder(radius=0.0015, segment=points)
            line.visual.vertex_colors = pts_colors[idx].astype(np.uint8)
            scene.add(pyrender.Mesh.from_trimesh(line))

            box = trimesh.creation.box(extents=extents,
                                       transform=trimesh.transformations.translation_matrix(intersection_point))
            box.visual.vertex_colors = pts_colors[idx].astype(np.uint8)
            scene.add(pyrender.Mesh.from_trimesh(box))


    # Define camera behind anchor image
    camera_pose = np.array([
        [1, 0, 0, cam_offset_x],
        [0, 1, 0, cam_offset_y],
        [0, 0, 1, cam_offset_z],
        [0, 0, 0, 1]
    ])

    # Compute the rotation matrix
    rotation = Rotation.from_rotvec(angle_y * (np.pi / 180) * np.array([0, 1, 0])).as_matrix()

    # Update the camera pose
    camera_pose[:3, :3] = rotation @ camera_pose[:3, :3]
    camera_pose[:3, 3] = rotation @ (camera_pose[:3, 3] - center) + center

    # Compute the rotation matrix
    rotation = Rotation.from_rotvec(angle_x * (np.pi / 180) * np.array([1, 0, 0])).as_matrix()

    # Update the camera pose
    camera_pose[:3, :3] = rotation @ camera_pose[:3, :3]
    camera_pose[:3, 3] = rotation @ (camera_pose[:3, 3] - center) + center

    # Define camera and lights
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose_OpenGL = camera_pose @ OPENGL

    # Update the camera and light poses in the scene
    scene.add(camera, pose=camera_pose_OpenGL)

    r = pyrender.OffscreenRenderer(1000, 720)
    render_3d, _ = r.render(scene)

    img0 = 255 * data['image0'].permute(0, 2, 3, 1)[batch_id].detach().cpu().numpy()
    img1 = 255 * data['image1'].permute(0, 2, 3, 1)[batch_id].detach().cpu().numpy()

    # Define the thickness of the border
    border_size = 15

    # Add the border
    img0 = cv2.copyMakeBorder(img0, border_size, border_size, border_size, border_size,
                                        cv2.BORDER_CONSTANT, value=color_src_frame)
    img1 = cv2.copyMakeBorder(img1, border_size, border_size, border_size, border_size,
                                        cv2.BORDER_CONSTANT, value=color_dst_frame)

    # Downsample image as a visual hint
    factor_down = 2.5
    new_size = (int(img0.shape[1]/factor_down), int(img0.shape[0]/factor_down))
    img0 = cv2.resize(img0, new_size, interpolation=cv2.INTER_LINEAR)
    img1 = cv2.resize(img1, new_size, interpolation=cv2.INTER_LINEAR)

    offset = (render_3d.shape[0] - 2 * new_size[1] - 50)//2
    offset_w = 100
    template = 255*np.ones((render_3d.shape[0], render_3d.shape[1] + new_size[0]+offset_w, 3))
    template[:, new_size[0]+offset_w:] = render_3d
    template[offset:offset+new_size[1], 20:20+new_size[0]] = img0
    template[50+offset+new_size[1]:50+offset+2*new_size[1], 20:20+new_size[0]] = img1

    # Add confidence
    if add_confidence:
        conf_pose = min(len(point_cloud[0]) / (total_matches * max_conf_th), 1.0)
        # Create an array with the cool colormap
        height_box, width_box = 30, 200
        image = np.zeros((height_box, width_box))
        for i in range(width_box):
            image[:, i] = (width_box-i) / width_box

        # Convert the colormap to BGR format and scale to [0, 255]
        image_rgb = plt.cm.cool(image)[:, :, :3]  # Get the RGB values from the colormap
        image_rgb = (image_rgb * 255).astype(np.uint8)  # Scale to [0, 255]

        # Add confidence value
        conf = int(width_box * conf_pose)
        image_rgb[:, conf:] = 255

        # Create a black border
        border_size = 2
        image_rgb[:border_size, :] = 0
        image_rgb[-border_size:, :] = 0
        image_rgb[:, :border_size] = 0
        image_rgb[:, -border_size:] = 0

        template[50 + offset + 2 * new_size[1] - height_box: 50 + offset + 2 * new_size[1], 20 + new_size[0] + 50:20 + new_size[0] + 50 + width_box] = image_rgb

        # Add text to the image
        cv2.putText(template, 'Confidence', (20 + new_size[0] + 50, 50 + offset + 2 * new_size[1] - height_box - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Add reference information
    height_box, width_box = 20, 35
    box_color = np.ones((height_box, width_box, 3))
    box_color = box_color * color_src_frame[:3]

    # Create a border
    border_size = 2
    box_color[:border_size, :] = 0
    box_color[-border_size:, :] = 0
    box_color[:, :border_size] = 0
    box_color[:, -border_size:] = 0

    template[50 + offset + 2 * new_size[1] - height_box - 40 * 3: 50 + offset + 2 * new_size[1] - 40 * 3,
    20 + new_size[0] + 50:20 + new_size[0] + 50 + width_box] = box_color

    # Add text to the image
    cv2.putText(template, 'Reference', (20 + new_size[0] + 50 + width_box + 10,
                        50 + offset + 2 * new_size[1] - 40 * 3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Add destination information
    height_box, width_box = 20, 35
    box_color = np.ones((height_box, width_box, 3))
    box_color = box_color * color_dst_frame[:3]

    # Create a black border
    border_size = 2
    box_color[:border_size, :] = 0
    box_color[-border_size:, :] = 0
    box_color[:, :border_size] = 0
    box_color[:, -border_size:] = 0

    template[50 + offset + 2 * new_size[1] - height_box - 40 * 2: 50 + offset + 2 * new_size[1] - 40 * 2,
    20 + new_size[0] + 50:20 + new_size[0] + 50 + width_box] = box_color

    # Add text to the image
    cv2.putText(template, 'Destination', (20 + new_size[0] + 50 + width_box + 10,
                                          50 + offset + 2 * new_size[1] - 40 * 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    return template

