import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time

import rclpy.subscription
from rosidl_runtime_py import message_to_ordereddict
import message_filters as mf

import tf2_ros as tf2

from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo, Image, Imu

import cv2
from cv_bridge import CvBridge
import json
import numpy as np
import os


class DataSaverNode(Node):
    def __init__(self):
        super().__init__("data_saver_node")

        self.declare_parameters(
            namespace="",
            parameters=[
                ("save_path", "/root/Documents/data/data0"),
                ("left_camera_info_topic", "/front_stereo_camera/left/camera_info"),
                ("left_camera_color_topic", "/front_stereo_camera/left/image_raw"),
                ("left_camera_depth_topic", "/front_stereo_camera/left/depth"),
                ("right_camera_info_topic", "/front_stereo_camera/right/camera_info"),
                ("right_camera_color_topic", "/front_stereo_camera/right/image_raw"),
                ("right_camera_depth_topic", "/front_stereo_camera/right/depth"),
                ("imu_topic", "/front_stereo_imu/imu"),
                ("gt_ref_frame_id", "World"),
            ]
        )

        self.save_path = self.get_parameter("save_path").value
        self.left_camera_info_topic = self.get_parameter("left_camera_info_topic").value
        self.left_camera_color_topic = self.get_parameter("left_camera_color_topic").value
        self.left_camera_depth_topic = self.get_parameter("left_camera_depth_topic").value
        self.right_camera_info_topic = self.get_parameter("right_camera_info_topic").value
        self.right_camera_color_topic = self.get_parameter("right_camera_color_topic").value
        self.right_camera_depth_topic = self.get_parameter("right_camera_depth_topic").value
        self.imu_topic = self.get_parameter("imu_topic").value
        self.gt_ref_frame_id = self.get_parameter("gt_ref_frame_id").value

        self.COLOR_DIR = "/color"
        self.DEPTH_DIR = "/depth"
        self.IMU_DIR = "/imu"
        self.TF_DIR = "/tf"

        self.counter = 0
        self.left_camera_info = None
        self.right_camera_info = None
        self.left_image_color = None
        self.left_image_depth = None
        self.right_image_color = None
        self.right_image_depth = None
        self.imu_data = None
        self.depth_scale = 1000.0

        self.left_camera_info_subscription = self.create_subscription(
            msg_type=CameraInfo,
            topic=self.left_camera_info_topic,
            callback=self.left_camera_info_callback,
            qos_profile=10,
        )

        self.right_camera_info_subscription = self.create_subscription(
            msg_type=CameraInfo,
            topic=self.right_camera_info_topic,
            callback=self.right_camera_info_callback,
            qos_profile=10,
        )

        self.left_color_sub = mf.Subscriber(self, Image, self.left_camera_color_topic)
        self.left_depth_sub = mf.Subscriber(self, Image, self.left_camera_depth_topic)
        self.right_color_sub = mf.Subscriber(self, Image, self.right_camera_color_topic)
        self.right_depth_sub = mf.Subscriber(self, Image, self.right_camera_depth_topic)
        self.imu_sub = mf.Subscriber(self, Imu, self.imu_topic)
        self.ats = mf.ApproximateTimeSynchronizer(
            [
                self.left_color_sub,
                self.left_depth_sub,
                self.right_color_sub,
                self.right_depth_sub,
                self.imu_sub,
            ],
            queue_size=10,
            slop=0.01,
        )
        self.ats.registerCallback(self.synced_callback)

        # TF2
        self.tf_buffer = tf2.Buffer(node=self)
        self.tf_listener = tf2.TransformListener(self.tf_buffer, self, spin_thread=True, qos=10)

        # Timer
        self.timer = self.create_timer(0.2, self.timer_callback)

        self.get_logger().info("Data Saver Node has been initialized.")

    def generate_undistort_map(self, camera_info: CameraInfo):
        camera_matrix = np.array(np.reshape(camera_info.k, (3, 3)))
        camera_distortion = np.array(camera_info.d)
        h, w = camera_info.height, camera_info.width
        # Define the new camera matrix and ROI
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            camera_distortion,
            (w, h),
            0.0,
            (w, h)
        )

        # Initialize undistort rectify map
        mapx, mapy = cv2.initUndistortRectifyMap(
            camera_matrix,
            camera_distortion,
            None,
            new_camera_matrix,
            (w, h),
            5 # CV_32FC1
        )

        return new_camera_matrix, mapx, mapy

    def left_camera_info_callback(self, msg: CameraInfo):
        if self.left_camera_info is not None:
            # We have already received the camera info.
            # Stop the subscription
            # del self.left_camera_info_subscription
            return

        self.left_camera_info = msg

        data_path = self.save_path

        if not os.path.exists(data_path):
            print(f'Creating new directory for data at "{data_path}".')
            os.mkdir(data_path)
        else:
            print("Data directory already exists. Cleaning up left side data...")
            os.system(f'rm -rf "{data_path}/left"')
            os.system(f'rm -rf "{data_path}{self.IMU_DIR}"')

        # Create the directories
        os.mkdir(f"{data_path}/left")
        os.mkdir(f"{data_path}/left{self.COLOR_DIR}")
        os.mkdir(f"{data_path}/left{self.DEPTH_DIR}")
        os.mkdir(f"{data_path}{self.IMU_DIR}")
        os.mkdir(f"{data_path}/left{self.TF_DIR}")

        with open(f"{data_path}/left/camera_info.json", "w") as f:
            json.dump(message_to_ordereddict(msg), f, indent=4)

        new_camera_matrix, self.left_mapx, self.left_mapy = self.generate_undistort_map(
            msg
        )

        # Save the new camera matrix and ROI to a JSON file
        udist_camera_matrix = {
            "camera_matrix": new_camera_matrix.tolist(),
            "width": self.left_camera_info.width,
            "height": self.left_camera_info.height,
            "depth_scale": self.depth_scale,
        }
        with open(f"{data_path}/left/udist_camera_matrix.json", "w") as f:
            json.dump(udist_camera_matrix, f, indent=4)

        self.get_logger().info("Left camera info has been saved.")

    def right_camera_info_callback(self, msg: CameraInfo):
        if self.right_camera_info is not None:
            # We have already received the camera info.
            # Stop the subscription
            # del self.right_camera_info_subscription
            return

        self.right_camera_info = msg

        data_path = self.save_path

        if not os.path.exists(data_path):
            print(f'Creating new directory for data at "{data_path}".')
            os.mkdir(data_path)
        else:
            print("Data directory already exists. Cleaning up right side data...")
            os.system(f'rm -rf "{data_path}/right"')

        # Create the directories
        os.mkdir(f"{data_path}/right")
        os.mkdir(f"{data_path}/right{self.COLOR_DIR}")
        os.mkdir(f"{data_path}/right{self.DEPTH_DIR}")
        # os.mkdir(f"{data_path}{self.IMU_DIR}")
        os.mkdir(f"{data_path}/right{self.TF_DIR}")

        with open(f"{data_path}/right/camera_info.json", "w") as f:
            json.dump(message_to_ordereddict(msg), f, indent=4)

        new_camera_matrix, self.right_mapx, self.right_mapy = (
            self.generate_undistort_map(msg)
        )

        # Save the new camera matrix and ROI to a JSON file
        udist_camera_matrix = {
            "camera_matrix": new_camera_matrix.tolist(),
            "width": self.right_camera_info.width,
            "height": self.right_camera_info.height,
            "depth_scale": self.depth_scale,
        }
        with open(f"{data_path}/right/udist_camera_matrix.json", "w") as f:
            json.dump(udist_camera_matrix, f, indent=4)

        self.get_logger().info("Right camera info has been saved.")

    def synced_callback(
        self,
        left_color: Image,
        left_depth: Image,
        right_color: Image,
        right_depth: Image,
        imu: Imu,
    ):
        self.left_image_color = left_color
        self.left_image_depth = left_depth
        self.right_image_color = right_color
        self.right_image_depth = right_depth
        self.imu_data = imu

    def timer_callback(self):
        # Save the latest data to files

        if (self.left_camera_info is None) or (self.right_camera_info is None):
            return

        if any(
            arg is None
            for arg in [
                self.left_image_color,
                self.left_image_depth,
                self.right_image_color,
                self.right_image_depth,
                self.imu_data,
            ]
        ):
            return

        bridge = CvBridge()

        cv_image = bridge.imgmsg_to_cv2(self.left_image_color)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        udist_image = cv2.remap(cv_image, self.left_mapx, self.left_mapy, cv2.INTER_LINEAR)
        cv2.imwrite(f"{self.save_path}/left{self.COLOR_DIR}/{self.counter:06d}.png", udist_image)

        cv_image = bridge.imgmsg_to_cv2(self.left_image_depth)
        cv_image_mm = cv_image * self.depth_scale
        udist_image = cv2.remap(cv_image_mm, self.left_mapx, self.left_mapy, cv2.INTER_NEAREST)
        udist_image = udist_image.astype(np.uint16)
        cv2.imwrite(f"{self.save_path}/left{self.DEPTH_DIR}/{self.counter:06d}.png", udist_image)


        cv_image = bridge.imgmsg_to_cv2(self.right_image_color)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        udist_image = cv2.remap(cv_image, self.right_mapx, self.right_mapy, cv2.INTER_LINEAR)
        cv2.imwrite(f"{self.save_path}/right{self.COLOR_DIR}/{self.counter:06d}.png", udist_image)

        cv_image = bridge.imgmsg_to_cv2(self.right_image_depth)
        cv_image_mm = cv_image * self.depth_scale
        udist_image = cv2.remap(cv_image_mm, self.right_mapx, self.right_mapy, cv2.INTER_NEAREST)
        udist_image = udist_image.astype(np.uint16)
        cv2.imwrite(f"{self.save_path}/right{self.DEPTH_DIR}/{self.counter:06d}.png", udist_image)

        world_to_left = TransformStamped()
        try:
            world_to_left = self.tf_buffer.lookup_transform(
                target_frame=self.gt_ref_frame_id,
                source_frame=self.left_image_color.header.frame_id,
                time=Time.from_msg(self.left_image_color.header.stamp)
            )
        except Exception as e:
            self.get_logger().error(f"Could not get transform: {e}")

        left_tf = message_to_ordereddict(world_to_left)
        with open(f"{self.save_path}/left{self.TF_DIR}/{self.counter:06d}_tf.json", "w") as f:
            json.dump(left_tf, f, indent=4)

        world_to_right = TransformStamped()
        try:
            world_to_right = self.tf_buffer.lookup_transform(
                target_frame=self.gt_ref_frame_id,
                source_frame=self.right_camera_info.header.frame_id,
                time=Time.from_msg(self.right_image_depth.header.stamp)
            )
        except Exception as e:
            self.get_logger().error(f"Could not get transform: {e}")

        right_tf = message_to_ordereddict(world_to_right)
        with open(f"{self.save_path}/right{self.TF_DIR}/{self.counter:06d}_tf.json", "w") as f:
            json.dump(right_tf, f, indent=4)

        # IMU data
        imu_data = message_to_ordereddict(self.imu_data)
        with open(f"{self.save_path}{self.IMU_DIR}/{self.counter:06d}.json", "w") as f:
            json.dump(imu_data, f, indent=4)

        # IMU ground-truth transform
        world_to_imu = TransformStamped()
        try:
            world_to_imu = self.tf_buffer.lookup_transform(
                target_frame=self.gt_ref_frame_id,
                source_frame=self.imu_data.header.frame_id,
                time=Time.from_msg(self.imu_data.header.stamp)
            )
        except Exception as e:
            self.get_logger().error(f"Could not get transform: {e}")

        imu_tf = message_to_ordereddict(world_to_imu)
        with open(f"{self.save_path}{self.IMU_DIR}/{self.counter:06d}_tf.json", "w") as f:
            json.dump(imu_tf, f, indent=4)

        self.get_logger().info(f"Saved left and right camera data: {self.counter:06d}.")
        self.counter += 1


def main(args=None):
    rclpy.init(args=args)
    node = DataSaverNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
