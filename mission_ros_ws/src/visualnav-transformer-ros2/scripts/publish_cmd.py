from typing import Tuple

import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from visualnav_transformer.deployment.topic_names import WAYPOINT_TOPIC

# CONSTS
CONFIG_PATH = "config/robot.yaml"
with open(CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
VEL_TOPIC = robot_config["vel_navi_topic"]
DT = 1 / robot_config["frame_rate"]
RATE = 9
EPS = 1e-8
WAYPOINT_TIMEOUT = 1  # seconds # TODO: tune this
FLIP_ANG_VEL = np.pi / 4
LINEAR_VEL = 1e-2


def pd_controller(waypoint: np.ndarray) -> Tuple[float]:
    """PD controller for the robot"""
    assert (
        len(waypoint) == 2 or len(waypoint) == 4
    ), "waypoint must be a 2D or 4D vector"
    if len(waypoint) == 2:
        dx, dy = waypoint
    else:
        dx, dy, hx, hy = waypoint
    # this controller only uses the predicted heading if dx and dy near zero
    if len(waypoint) == 4 and np.abs(dx) < EPS and np.abs(dy) < EPS:
        v = 0
        w = clip_angle(np.arctan2(hy, hx)) / DT
    elif np.abs(dx) < EPS:
        v = 0
        w = np.sign(dy) * np.pi / (2 * DT)
    else:
        v = LINEAR_VEL * dx / np.abs(dy)
        w = np.arctan(dy / dx)
    v = np.clip(v, -MAX_V, MAX_V)
    w = np.clip(w, -MAX_W, MAX_W)
    return v, w


class VelPublisher(Node):
    def __init__(self):
        super().__init__("cmd_vel_publisher")

        self.waypoint_subscription = self.create_subscription(
            Float32MultiArray, WAYPOINT_TOPIC, self.waypoint_callback, 10
        )

        # Publisher
        self.publisher = self.create_publisher(Twist, "/cmd_vel", 10)

        self.bridge = CvBridge()
        self.latest_image = None
        self.latest_waypoint = None

    def waypoint_callback(self, msg):
        v, w = pd_controller(msg.data)
        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w
        self.publisher.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = VelPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
