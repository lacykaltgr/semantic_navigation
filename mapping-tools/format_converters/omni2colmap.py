import glob
import argparse
import shutil
from pathlib import Path
import os
import json
import numpy as np

from colmap_utils import Camera, Image, write_model


def pose_matrix_from_quaternion(position, orientation):
    """ convert 4x4 pose matrix to (t, q) """
    from scipy.spatial.transform import Rotation

    pose = np.eye(4)
    rot = Rotation.from_quat(orientation).as_matrix()
    pose[:3, :3] = rot
    pose[:3, 3] = position
    #pose = np.linalg.inv(pose) # TODO: remove??
    return pose

def quaternion_from_pose_matrix(pose):
    """ convert (t, q) to 4x4 pose matrix """
    from scipy.spatial.transform import Rotation

    rot = pose[:3, :3]
    rot = Rotation.from_matrix(rot).as_quat()
    x, y, z, w = rot
    rotation = np.array([w, x, y, z])

    trans = pose[:3, 3]
    x, y, z = trans
    translation = np.array([x, y, z])
    return translation, rotation


def omni2colmap(omni_dir_path, colmap_dir_path):
    colmap_cameras = {}
    colmap_images = {}
    colmap_points3D = {}

    omni_images_path = os.path.join(omni_dir_path, 'color')
    omni_camera_info_path = os.path.join(omni_dir_path, 'udist_camera_matrix.json')
    omni_transforms_path = os.path.join(omni_dir_path, 'tf')
    colmap_images_path = os.path.join(colmap_dir_path, 'images')

    with open(omni_camera_info_path, 'r') as f:
        omni_transforms = json.load(f)
        camera_matrix = np.array(omni_transforms['camera_matrix'])
        width = omni_transforms['width']
        height = omni_transforms['height']
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
    colmap_camera_id = 1 # only works for 1 camera for now
    camera_dict = {
        'model': 'PINHOLE',
        'width': width,
        'height': height,
        'params': [fx, fy, cx, cy]
    }
    colmap_cameras[colmap_camera_id] = Camera(colmap_camera_id, **camera_dict)

    # copy all images from omni images path to colmap images path
    os.makedirs(colmap_images_path, exist_ok=True)
    colmap_image_paths = []
    image_files = sorted(glob.glob(os.path.join(omni_images_path, '*.png')))
    for i, image_file in enumerate(image_files):
        image_path = f"{i:06d}.png"
        colmap_image_paths.append(image_path)
        shutil.copy(image_file, os.path.join(colmap_images_path, image_path))

    # loop thought transforms dir
    # files are name 000000_tf.json, 000001_tf.json, etc
    transform_files = glob.glob(os.path.join(omni_transforms_path, '*.json'))
    transform_files = sorted(transform_files, key=lambda x: int(os.path.basename(x).split('_')[0]))
    for i, transform_file in enumerate(transform_files):
        with open(transform_file, 'r') as f:
            transform = json.load(f)['transform']
            rot = transform['rotation']
            trans = transform['translation']
            # construct 4x4 transformation matrix
            rotation = np.array([rot['x'], rot['y'], rot['z'], rot['w']])
            translation = np.array([trans['x'], trans['y'], trans['z']])
            pose = pose_matrix_from_quaternion(translation, rotation)
            # take the inverse of the pose matrix w2c -> c2w
            pose = np.linalg.inv(pose)
            tvec, qvec = quaternion_from_pose_matrix(pose)
            image_path = colmap_image_paths[i]
            colmap_images[i] = Image(i, qvec, tvec, colmap_camera_id, image_path, [], [])

    return colmap_cameras, colmap_images, colmap_points3D


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Omniverse record to Colmap converter",
        description="This script converts an existing omniverse dataset into Colmap format",
    )
    parser.add_argument("--input_omni_dir", type=Path, required=True)
    parser.add_argument("--output_colmap_dir", type=Path, required=True)
    parser.add_argument("--ext", type=str, default='.bin')
    args = parser.parse_args()
    print("Input Omniverse dir: " + str(args.input_omni_dir))
    print("Output Colmap dir: " + str(args.output_colmap_dir))
    print("Output Colmap extension: " + str(args.ext))

    colmap_cameras, colmap_images, colmap_points3D = omni2colmap(args.input_omni_dir, args.output_colmap_dir)
    print("Writing Colmap model to " + str(args.output_colmap_dir))
    write_model(colmap_cameras, colmap_images, colmap_points3D, args.output_colmap_dir, args.ext)


