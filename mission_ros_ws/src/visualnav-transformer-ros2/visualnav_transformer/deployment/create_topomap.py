import argparse
import os
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Joy
from visualnav_transformer.deployment.utils import msg_to_pil

from visualnav_transformer.deployment.topic_names import IMAGE_TOPIC
TOPOMAP_IMAGES_DIR = "topomaps/images"
obs_img = None

def remove_files_in_dir(dir_path: str):
    for f in os.listdir(dir_path):
        file_path = os.path.join(dir_path, f)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))

def callback_obs(msg: Image):
    global obs_img
    obs_img = msg_to_pil(msg)

class CreateTopomapNode(Node):
    def __init__(self, args_dir, args_dt):
        super().__init__("CREATE_TOPOMAP")
        self.image_subscription = self.create_subscription(Image, IMAGE_TOPIC, callback_obs, 10)
        self.topomap_name_dir = os.path.join(TOPOMAP_IMAGES_DIR, args_dir)
        if not os.path.isdir(self.topomap_name_dir):
            os.makedirs(self.topomap_name_dir)
        else:
            print(f"{self.topomap_name_dir} already exists. Removing previous images...")
            remove_files_in_dir(self.topomap_name_dir)
        assert args_dt > 0, "dt must be positive"
        self.timer = self.create_timer(args_dt, self.run)
        print("Registered with master node. Waiting for images...")
        self.i = 0
        self.start_time = float("inf")

    def run(self):
        global obs_img
        if obs_img is not None:
            obs_img.save(os.path.join(self.topomap_name_dir, f"{self.i}.png"))
            print("published image", self.i)
            self.i += 1
            self.start_time = time.time()
            obs_img = None
        else:
            print(f"No image received.")

def main(args_dir="topomap", args_dt=3.0):
    rclpy.init()
    node = CreateTopomapNode(args_dir, args_dt)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Code to generate topomaps from the {IMAGE_TOPIC} topic"
    )
    parser.add_argument(
        "--dir",
        "-d",
        default="topomap",
        type=str,
        help="path to topological map images in ../topomaps/images directory (default: topomap)",
    )
    parser.add_argument(
        "--dt",
        "-t",
        default=1.0,
        type=float,
        help=f"time between images sampled from the {IMAGE_TOPIC} topic (default: 3.0)",
    )
    args = parser.parse_args()

    main(args)
