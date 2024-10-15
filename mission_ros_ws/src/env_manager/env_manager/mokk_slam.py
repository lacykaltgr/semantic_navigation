## mock slam node
# subscribes to /env/observation topic
# extracts position and orientation from the message
# publishes the position and orientation as tf_ros2 message

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped
import tf2_ros
import tf2_geometry_msgs
from std_msgs.msg import String
import json

class MokkSlam(Node):
    def __init__(self):
        super().__init__('mokk_slam')
        self.subscription = self.create_subscription(
            String,
            '/env/observation',
            self.listener_callback,
            10
        )
        self.subscription  # prevent unused variable warning
        self.br = tf2_ros.TransformBroadcaster(self)
        self.get_logger().info("Mokk SLAM node has been initialized.")

    def listener_callback(self, msg):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'

        msg_json = json.loads(msg.data)
        coordinates = msg_json["position"][0] # [x, y, z, w, x, y, z] for 1st environment!

        t.transform.translation.x = coordinates[0]
        t.transform.translation.y = coordinates[1]
        t.transform.translation.z = coordinates[2]
        t.transform.rotation.w = coordinates[3]
        t.transform.rotation.x = coordinates[4]
        t.transform.rotation.y = coordinates[5]
        t.transform.rotation.z = coordinates[6]
        
        self.get_logger().info(f"transform: {t}")
        self.br.sendTransform(t)    

def main(args=None):
    rclpy.init(args=args)
    mokk_slam = MokkSlam()
    rclpy.spin(mokk_slam)
    mokk_slam.destroy_node()
    rclpy.shutdown()