import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import Image

from visualnav_transformer.deployment.src.topic_names import IMAGE_TOPIC


class ImagePublisherNode(Node):
    def __init__(self):
        super().__init__("image_publisher")
        self.publisher_ = self.create_publisher(Image, IMAGE_TOPIC, 10)
        self.timer = self.create_timer(1.0, self.publish_image)
        self.bridge = CvBridge()

    def publish_image(self):
        # Create a sample PIL image (replace this with your actual image)
        pil_image = PILImage.open("branch_on_the_path_2.png")

        # Convert PIL image to OpenCV format
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Convert OpenCV image to ROS Image message
        ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")

        # Publish the image
        self.publisher_.publish(ros_image)
        self.get_logger().info("Publishing image")


def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
