import rclpy
from rclpy.node import Node
import rclpy.subscription
import json 

from omni.isaac.lab.app import AppLauncher
app_launcher = AppLauncher(livestream=1)
simulation_app = app_launcher.app

from sensor_msgs.msg import Image
from std_msgs.msg import String
from .wrapper import GymIsaacWrapperForROSEnv
from .hworldmodel.isaac.envs import *
from .utils import space_to_dict, dict_np2list
import numpy as np

class EnvManager(Node):
    def __init__(self):
        super().__init__("env_stepper_node")

        self.action_topic = "/env/action"
        self.observation_topic = "/env/observation"
        self.step_info_topic = "/env/step_info" # reward, terminated, truncated
        self.env_info_topic = "/env/info"
        self.env_name = "Mokk-Navigation-v0"

        self.simulation_app = simulation_app
        self.env = self._init_env(self.env_name)

        self.action_subscription = self.create_subscription(
            msg_type=String,
            topic=self.action_topic,
            callback=self.action_callback,
            qos_profile=10,
        )
        self.observation_publisher = self.create_publisher(
            msg_type=String,
            topic=self.observation_topic,
            qos_profile=10,
        )
        self.step_info_publisher = self.create_publisher(
            msg_type=String,
            topic=self.step_info_topic,
            qos_profile=10,
        )
        self.env_step_timer = self.create_timer(
            timer_period_sec=0.4,  # TODO: set this according to the simulation frequency
            callback=self.env_step_timer_cb
        )
        self.env_info_publisher = self.create_publisher(
            msg_type=String,
            topic=self.env_info_topic,
            qos_profile=10,
        )

        self.reset_subscriber = self.create_subscription(
            msg_type=String,
            topic="/env/reset",
            callback=self.reset_callback,
            qos_profile=10,
        )

        self.action_received = False
        self.current_action = None
        self.should_reset = True
        self.last_obs_msg = None

        self.get_logger().info("Env Stepper Node has been initialized.")

    def reset_callback(self, msg):
        self.should_reset = True


    def action_callback(self, msg):
        action_string = msg.data
        action_json = json.loads(action_string)
        self.current_action = np.array([np.array(action_json["action"])])
        self.action_received = True

    def env_step_timer_cb(self):
        if not self.simulation_app.is_running():
            self.get_logger().info("Simulation has stopped. Exiting.")
            self.destroy_node()
            rclpy.shutdown()
            return
        
        if self.should_reset:
            obs_space_dict = space_to_dict(self.env.single_observation_space)
            act_space_dict = space_to_dict(self.env.single_action_space)
            env_info_msg = json.dumps({
                "num_envs": self.env.num_envs,
                "single_observation_space": obs_space_dict,
                "single_action_space": act_space_dict,
            })
            msg = String()
            msg.data = env_info_msg
            self.env_info_publisher.publish(msg)
            self.get_logger().info("Environment info published.")

            # wait a bit for action server to load policy?

            obs, _ = self.env.reset()
            obs = dict_np2list(obs)
            obs_msg = json.dumps(obs)
            msg = String()
            msg.data = obs_msg
            self.last_obs_msg = obs_msg
            self.observation_publisher.publish(msg)
            self.get_logger().info("Environment observation published.")
            self.should_reset = False
            return 
        
        if self.action_received:
            obs, rew, terminated, truncated, info = self.env.step(self.current_action)

            self.get_logger().info(f"Stepped env with action: {self.current_action}")

            obs_msg = dict_np2list(obs)
            obs_msg = json.dumps(obs_msg)

            step_info_msg = json.dumps({
                "reward": float(rew[0]),
                "terminated": bool(terminated[0]),
                "truncated": bool(truncated[0]),
                "info": dict_np2list(info),
            })

            msg_obs = String()
            self.last_obs_msg = obs_msg
            msg_obs.data = obs_msg
            msg_info = String()
            msg_info.data = step_info_msg

            self.observation_publisher.publish(msg_obs)
            self.step_info_publisher.publish(msg_info)

            self.get_logger().info("Environment observation published.")

            self.action_received = False
        else:
            if self.last_obs_msg is not None:
                msg_obs = String()
                msg_obs.data = self.last_obs_msg
                self.observation_publisher.publish(msg_obs)
                self.get_logger().info("No step observation published.")
            self.get_logger().info("Cannot step env: no action received.")


    def _init_env(self, env_name, num_envs=1):
        """
        Initialize the TorchRL environment.
        """
        assert num_envs == 1, "Only one environment is supported for now."

        import gymnasium as gym
        from omni.isaac.lab_tasks.utils import parse_env_cfg
        env_cfg = parse_env_cfg(env_name, use_gpu=True, num_envs=num_envs)
        env = gym.make(env_name, cfg=env_cfg)
        return GymIsaacWrapperForROSEnv(env)
    
    def reset_env(self):
        self.env.reset()


def main(args=None):
    rclpy.init(args=args)
    node = EnvManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
