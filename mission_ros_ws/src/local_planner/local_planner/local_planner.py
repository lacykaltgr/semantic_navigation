import rclpy
from rclpy.node import Node
import rclpy.subscription

from sensor_msgs.msg import Image
#from utils import make_env
#from utils import import_module_by_path
import yaml
import torch
from rclpy.action import ActionServer
import tf2_ros
from mission_planner_interfaces.action import MoveTo
from mission_planner_interfaces.msg import String
import numpy as np
import json
import os
from omegaconf import OmegaConf
from tf2_ros.transform_listener import TransformListener

from .utils import wrap_to_pi, heading_w, yaw_quat, quat_rotate_inverse
import time
from rclpy.executors import MultiThreadedExecutor


class LocalPlanner(Node):
    def __init__(self):
        super().__init__("local_planner")

        #self.vector_env = None
        #self.torchrl_env = None

        #self.declare_parameter("policy_dict_path", "/workspace/policy")

        #self.policy_dict_path = self.get_parameter("policy_dict_path").value
        #self.policy_state_dict_path = os.path.join(self.policy_dict_path, "policy.pth")
        #self.config_path = os.path.join(self.policy_dict_path, "config.yaml")

        #self.policy_state_dict = torch.load(self.policy_state_dict_path)
        #self.config = OmegaConf.load(open(self.config_path, "r"))
        #self.module_path = self.config.logger.package_path
        #self.module_name = self.config.logger.model_name

        #self.device = self.config.logger.device

        self.action_topic = "/env/action"
        self.observation_topic = "/env/observation"

        self.action_server = ActionServer(
            self,
            MoveTo,
            "/local_planner/move_to",
            self.command_callback,
        )

        self.action_publisher = self.create_publisher(
            msg_type=String,
            topic=self.action_topic,
            qos_profile=10,
        )
        self.observation_subscriber = self.create_subscription(
            msg_type=String,
            topic=self.observation_topic,
            callback=self.observation_callback,
            qos_profile=10,
        )

        #self.env_info_subscriber = self.create_subscription(
        #    msg_type=String,
        #    topic=self.observation_topic,
        #    callback=self.env_info_callback,
        #    qos_profile=10,
        #)

        self.tf_buffer = tf2_ros.Buffer(node=self)
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.should_reset = True
        # self.observation = None
        # self.state_td = None
        self.current_goal = None
        self.current_goal_radius = 0.0
        self.goal_handle = None
        self.feedback_msg = MoveTo.Feedback()
        self.is_goal_reached = False
        
        self.get_logger().info("Local Planner Node has been initialized.")
        self.counter = 0

    """
    def env_info_callback(self, msg):
        if self.torchrl_env is None:
            num_envs = msg.num_envs
            single_observation_space = msg.single_observation_space
            single_action_space = msg.single_action_space
            self.vector_env, self.torchrl_env = make_env(
                num_envs, single_observation_space, single_action_space, device=self.device
            )
            self.vector_env.set_dummy_observations()
            model_module = import_module_by_path(self.module_path, self.module_name)
            module = model_module(self.config, self.torchrl_env, self.device)
            policy = module.modules["policy"]
            policy.load_state_dict(self.policy_state_dict)
            self.policy = policy
            self.should_reset = True
    """

    def command_callback(self, goal_handle):
        goal_json = goal_handle.request.goal_position
        goal_json_dict = json.loads(goal_json)
        print(goal_json_dict)
        self.goal_handle = goal_handle
        self.current_goal = np.array([goal_json_dict["x"], goal_json_dict["y"], goal_json_dict["z"]])
        self.current_goal_radius = goal_json_dict["radius"]
        self.get_logger().info("Received goal: " + str(self.current_goal) + " with radius: " + str(goal_json_dict["radius"]))

        self.is_goal_reached = False
        
        while not self.is_goal_reached:
            time.sleep(0.5)
        
        # Now the task is done, mark the goal as succeeded
        self.get_logger().info("Goal reached.")
        goal_handle.succeed()

        self.current_goal = None
        self.current_goal_radius = 0.0
        self.goal_handle = None

        # Return the result
        result = MoveTo.Result()
        result.result = "ok"
        return result

    def observation_callback(self, msg):
        #if self.env is None:
         #    self.get_logger().info("Cannot use observation: env is not initialized.")
         #    return
        # self.vector_env.update_observations(msg)
        self.get_logger().info("Observation received.")
        self.step()

    def step(self):
        if self.current_goal is None:
            self.get_logger().info("Waiting for goal.")
            return

        if self.should_reset:
            self.get_logger().info("Resetting the environment.")
            #self.state_td = self.torchrl_env.reset()
            self.should_reset = False
            return
        
        #rollout = self.torchrl_env.rollout(
        #    max_steps=1,
        #    policy=self.policy,
        #    tensordict=self.state_td
        #)
        #self.state_td = rollout[-1]
        #action = self.state_td["action"][-1]
        #self.action_publisher.publish(action)

        position_tf = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
        position = torch.tensor([position_tf.transform.translation.x, 
                                position_tf.transform.translation.y,
                                0.0], dtype=torch.float32)
        goal = torch.tensor([self.current_goal[0], self.current_goal[1], 0.0])
        target_vec = goal - position


        target_direction = torch.atan2(target_vec[1], target_vec[0])
        flipped_target_direction = wrap_to_pi(target_direction + torch.pi)


        rotation_quat = torch.tensor([position_tf.transform.rotation.w, 
                                        position_tf.transform.rotation.x, 
                                        position_tf.transform.rotation.y, 
                                        position_tf.transform.rotation.z], 
                                        dtype=torch.float32)
        rotation = heading_w(torch.tensor(rotation_quat))
        curr_to_target = wrap_to_pi(target_direction - rotation).abs()
        curr_to_flipped_target = wrap_to_pi(flipped_target_direction - rotation).abs()


        if curr_to_target <  curr_to_flipped_target:
            heading_command = target_direction
        else:
            heading_command = flipped_target_direction

        pos_command_b = quat_rotate_inverse(yaw_quat(rotation_quat), target_vec)
        heading_command_b = wrap_to_pi(heading_command - rotation)
        # 2d action

        pos_command_b = pos_command_b[:2] / torch.max(torch.abs(pos_command_b[:2]))

        if heading_command_b.abs() > 3.14 / 4:
            command_b = torch.cat([torch.tensor([0.0, 0.0]), heading_command_b], dim=0)
        else:
            command_b = torch.cat([pos_command_b, heading_command_b], dim=0)

        # if length of vector is less than 0.1, goal is reached
        distance_to_goal = np.linalg.norm(target_vec.numpy(), ord=2)
        if distance_to_goal < self.current_goal_radius:
            self.get_logger().info("Goal reached.")
            self.is_goal_reached = True
            return

        self.feedback_msg.feedback = json.dumps({"distance_to_goal": float(distance_to_goal)})
        self.goal_handle.publish_feedback(self.feedback_msg)
        
        action_msg = String()
        action_msg.data = json.dumps({"action": command_b.numpy().tolist()})
        self.action_publisher.publish(action_msg)
        self.get_logger().info(f"Action published. Distance: {distance_to_goal}")

def main(args=None):
    rclpy.init(args=args)
    node = LocalPlanner()
    executor = MultiThreadedExecutor()
    rclpy.spin(node, executor=executor)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
