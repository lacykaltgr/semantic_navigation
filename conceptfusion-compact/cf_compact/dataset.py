
from typing import Optional
import os
import json
import glob
import numpy as np
import torch
from natsort import natsorted

from cf_compact.datautils.dataset_base import GradSLAMDataset


class ConceptFusionDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.colmap = False
        self.input_folder = basedir
        self.pose_paths = sorted(glob.glob(os.path.join(self.input_folder, "tf/*_tf.json")))
        
        camera_info_path = os.path.join(self.input_folder, "udist_camera_matrix.json")
        self.camera_info = json.load(open(camera_info_path, "r"))

        camera_params = dict(
            image_height=self.camera_info["height"],
            image_width=self.camera_info["width"],
            cx=self.camera_info["camera_matrix"][0][2],
            cy=self.camera_info["camera_matrix"][1][2],
            fx=self.camera_info["camera_matrix"][0][0],
            fy=self.camera_info["camera_matrix"][1][1],
            png_depth_scale=self.camera_info["depth_scale"],
            crop_edge=0,
        )
        super().__init__(
            {"camera_params": camera_params},
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/color/*.png"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(
                glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt")
            )
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        if self.colmap:
            return self.load_poses_from_colmap(
                os.path.join(self.input_folder, "images.txt")
            )
        poses = []
        for pose_path in self.pose_paths:
            pose = self.load_pose(pose_path)
            poses.append(pose)
        return poses

    def load_pose(self, path):
        pose_data = json.load(open(path, "r"))
        c2w_dict = pose_data["transform"]
        position = np.array(list(c2w_dict["translation"].values()))
        orientation = np.array(list(c2w_dict["rotation"].values()))
        pose = self.pose_matrix_from_quaternion(position, orientation)
        return torch.tensor(pose)

    def load_poses_from_colmap(self, path):
        pose_data = self.parse_colmap_images_txt(path)
        poses = []
        for image_data in pose_data:
            position = np.array(image_data["translation"])
            orientation = np.array(image_data["quaternion"])
            pose = self.pose_matrix_from_quaternion(position, orientation)
            poses.append(torch.tensor(pose))
        return poses
    
    def pose_matrix_from_quaternion(self, position, orientation):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        rot = Rotation.from_quat(orientation).as_matrix()
        pose[:3, :3] = rot
        pose[:3, 3] = position
        #pose = np.linalg.inv(pose) # TODO: remove??
        return pose

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)

    

    def parse_colmap_images_txt(file_path):
        import re
        images_data = []
        
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        current_image = None
        keypoint_pattern = re.compile(r"\d+\.\d+ \d+\.\d+")

        for line in lines:
            line = line.strip()

            # Skip comments or empty lines
            if line.startswith('#') or not line:
                continue

            parts = line.split()
            if len(parts) == 9:  # Quaternion and Translation + Filename
                if current_image:
                    images_data.append(current_image)  # Save the previous image data before processing the new one

                # Parse the image data
                image_id = int(parts[0])
                quaternion = list(map(float, parts[1:5]))  # q1, q2, q3, q4
                translation = list(map(float, parts[5:8]))  # tx, ty, tz
                camera_id = int(parts[8])
                image_filename = parts[9]
                
                current_image = {
                    'image_id': image_id,
                    'quaternion': quaternion,
                    'translation': translation,
                    'camera_id': camera_id,
                    'image_filename': image_filename,
                    'keypoints': []
                }
                images_data.append(current_image)

        return images_data