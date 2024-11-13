import os
import json
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


import os
import json
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass


# Function to convert COLMAP camera parameters to OpenCV format
def colmap_to_opencv_camera(camera_data):
    params = camera_data['params']
    width, height = camera_data['width'], camera_data['height']
    
    # For SIMPLE_RADIAL camera model:
    # params = [f, cx, cy, k1]
    camera_matrix = [
        params[0], 0.0, params[1],
        0.0, params[0], params[2],
        0.0, 0.0, 1.0
    ]
    
    # Extract distortion coefficient (k1)
    distortion = [params[3], 0.0, 0.0, 0.0, 0.0]  # k1, k2, p1, p2, k3
    
    return CameraInfo(
        k=camera_matrix,
        d=distortion,
        height=height,
        width=width
    )

# Utility functions for reading COLMAP files
def read_cameras_text(path):
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = {
                    'id': camera_id, 'model': model, 'width': width, 'height': height, 'params': params
                }
    return cameras

def read_images_text(path):
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                images[image_id] = {
                    'id': image_id, 'qvec': qvec, 'tvec': tvec,
                    'camera_id': camera_id, 'name': image_name
                }
    return images


# Function to convert COLMAP quaternion and translation to transformation dictionary
def colmap_to_transform(qvec, tvec):
    rotation = R.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]])  # COLMAP quaternion format is [qw, qx, qy, qz]
    translation = {
        'x': float(tvec[0]), 'y': float(tvec[1]), 'z': float(tvec[2])
    }
    rotation_dict = {
        'x': float(rotation.as_quat()[0]), 'y': float(rotation.as_quat()[1]),
        'z': float(rotation.as_quat()[2]), 'w': float(rotation.as_quat()[3])
    }
    return {'translation': translation, 'rotation': rotation_dict}


# Function to apply a 4x4 transformation matrix to a pose (quaternion and translation)
def apply_transformation(qvec, tvec, transformation_matrix):
    # Convert quaternion and translation to a 4x4 transformation matrix
    rotation = R.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]])
    rotation_matrix = rotation.as_matrix()
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = rotation_matrix
    pose_matrix[:3, 3] = tvec

    # take the inverse
    pose_matrix = np.linalg.inv(pose_matrix)

    # Apply the transformation
    global_pose_matrix = transformation_matrix @ pose_matrix

    # Extract the new rotation and translation
    new_rotation = R.from_matrix(global_pose_matrix[:3, :3])
    new_translation = global_pose_matrix[:3, 3]

    # Return the updated quaternion and translation
    new_qvec = new_rotation.as_quat()
    # Return as [qw, qx, qy, qz] format expected by COLMAP
    return [new_qvec[3], new_qvec[0], new_qvec[1], new_qvec[2]], new_translation

# Main conversion function
def convert_data(input_dir, output_dir, transformations):
    # Create output directories
    color_dir = os.path.join(output_dir, 'color')
    depth_dir = os.path.join(output_dir, 'depth')
    tf_local_dir = os.path.join(output_dir, 'tf_local')
    tf_dir = os.path.join(output_dir, 'tf')
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(tf_local_dir, exist_ok=True)
    os.makedirs(tf_dir, exist_ok=True)

    # Construct paths for camera and image files within the input directory
    camera_file = os.path.join(input_dir, "cameras.txt")
    images_file = os.path.join(input_dir, "images.txt")

    # Verify that the required files exist
    if not os.path.exists(camera_file):
        raise FileNotFoundError(f"cameras.txt not found in {input_dir}")
    if not os.path.exists(images_file):
        raise FileNotFoundError(f"images.txt not found in {input_dir}")

    # Read COLMAP data
    cameras = read_cameras_text(camera_file)
    images = read_images_text(images_file)


    # Determine transformation matrix for the current input directory
    input_label = os.path.basename(os.path.dirname(input_dir))  # Use the label from the parent directory
    transformation_matrix = None
    for item in transformations:
        if item["label"] == input_label:
            transformation_matrix = np.array(item["transformation"])
            break
    if transformation_matrix is None:
        raise ValueError(f"Transformation not found for label {input_label}")

    # Process each image
    from tqdm import tqdm
    for image_id, image_data in tqdm(images.items()):
        image_name = image_data['name']
        base_name = os.path.splitext(image_name)[0].split('_')[-1].zfill(6)  # Format as 000000

        # Downscale RGB image
        input_rgb_path = os.path.join(input_dir, 'cam0', image_name)
        image = cv2.imread(input_rgb_path)
        output_rgb_path = os.path.join(color_dir, f'{base_name}.png')
        cv2.imwrite(output_rgb_path, image)

        # Copy depth image
        input_depth_path = os.path.join(input_dir, 'depth', f'{base_name}.png')
        output_depth_path = os.path.join(depth_dir, f'{base_name}.png')
        depth_image = cv2.imread(input_depth_path, cv2.IMREAD_UNCHANGED)
        cv2.imwrite(output_depth_path, depth_image)

        # Save local transformation
        local_transform = colmap_to_transform(image_data['qvec'], image_data['tvec'])
        local_transform_file_path = os.path.join(tf_local_dir, f'{base_name}_tf.json')
        with open(local_transform_file_path, 'w') as f:
            json.dump({'transform': local_transform}, f, indent=4)

        # Apply transformation to get global pose
        global_qvec, global_tvec = apply_transformation(image_data['qvec'], image_data['tvec'], transformation_matrix)
        global_transform = colmap_to_transform(global_qvec, global_tvec)
        global_transform_file_path = os.path.join(tf_dir, f'{base_name}_tf.json')
        with open(global_transform_file_path, 'w') as f:
            json.dump({'transform': global_transform}, f, indent=4)

    # Write undistorted camera matrix
    camera_id = list(cameras.keys())[0]  # Assuming all images use the same camera
    camera_data = cameras[camera_id]
    params = camera_data['params']
    width, height = camera_data['width'], camera_data['height']
    camera_matrix = [
        [params[0], 0.0, params[2]],
        [0.0, params[1], params[3]],
        [0.0, 0.0, 1.0]
    ]
    camera_info = {
        'camera_matrix': camera_matrix,
        'width': width,
        'height': height,
        'depth_scale': 1000.0,
    }
    camera_output_path = os.path.join(output_dir, 'udist_camera_matrix.json')
    with open(camera_output_path, 'w') as f:
        json.dump(camera_info, f, indent=4)

    print(f"Conversion completed for {input_dir}.")

# Function to handle multiple directories
def convert_multiple_directories(input_dirs, output_dirs, transformation_file):
    # Read transformations from the JSON file
    with open(transformation_file, 'r') as f:
        transformations = json.load(f)

    for input_dir, output_dir in zip(input_dirs, output_dirs):
        # Perform conversion
        convert_data(input_dir, output_dir, transformations)

"""
data/skypark/6d643c9387
data/skypark/6fe4353ab4
data/skypark/8a4e709d40
data/skypark/61ea1c698a
data/skypark/73bb8cd3f3
data/skypark/253a30fef3
data/skypark/779a89c390
data/skypark/793ba655a6
data/skypark/99540c501d
data/skypark/ac5b204c1a
data/skypark/b7b4668bda
data/skypark/d787d601d2
data/skypark/e99243f6c6
data/skypark/f12b080cff
"""


# Example usage
labels = [
    #"6d643c9387",
    "6fe4353ab4",
    "8a4e709d40",
    #"61ea1c698a",
    "73bb8cd3f3",
    "253a30fef3",
    "779a89c390",
    "793ba655a6",
    "99540c501d",
    "ac5b204c1a",
    "b7b4668bda",
    "d787d601d2",
    "e99243f6c6",
    "f12b080cff",
]
input_directories = [
    f"/workspace/data/skypark/{label}/cvsl_recording" for label in labels
]
output_directories = [
    f"/workspace/data/skypark/{label}/omni" for label in labels
]
transformation_json = "/workspace/data/skypark/transforms.json"

convert_multiple_directories(input_directories, output_directories, transformation_json)