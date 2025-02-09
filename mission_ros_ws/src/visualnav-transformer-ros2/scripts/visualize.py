import matplotlib.pyplot as plt
import numpy as np
import rclpy
from sensor_msgs.msg import Image

from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

from visualnav_transformer.deployment.src.topic_names import (
    IMAGE_TOPIC,
    SAMPLED_ACTIONS_TOPIC,
)

data = []

img = np.zeros((100, 100, 3))
bridge = CvBridge()

def camera_callback(msg):
    global img
    img = bridge.imgmsg_to_cv2(msg)

def callback(msg):
    actions = msg.data
    actions = np.array(actions)[1:].reshape(-1, 2)
    actions = actions * img.shape[0] / 20
    actions[:, 1] = actions[:, 1] - img.shape[1] // 2
    actions[:, 0] = actions[:, 0] - img.shape[0]
    data = actions

    plt.gcf().canvas.draw()
    ax1.clear()
    ax1.imshow(img)

    for i in range(0, len(data), 8):
        ax1.plot(-data[i:i+8, 1], -data[i:i+8, 0]) 
    plt.pause(0.1)

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
plt.ion()
plt.show()

def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('sampled_actions_subscriber')
    camera_subscriber = node.create_subscription(Image, IMAGE_TOPIC, camera_callback, 10)
    subscriber = node.create_subscription(Float32MultiArray, SAMPLED_ACTIONS_TOPIC, callback, 1)
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()