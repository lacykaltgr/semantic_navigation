from rclpy.node import Node
import py_trees
from mission_planner_interfaces.action import MoveTo
from py_trees_ros.action_clients import FromBlackboard
import json
from std_msgs.msg import String
from . import READ, SUCCESS, FAILURE, WRITE, RUNNING


class WaypointHandler(py_trees.behaviour.Behaviour):
    def __init__(self, node):
        super(WaypointHandler, self).__init__(name="WaypointHandler")
        self.node = node

        # Attach blackboard
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key="path", access=READ)
        self.blackboard.register_key(key="next_waypoint", access=WRITE)

        # Internal state
        self.path = None
        self.path_length = 0
        self.waypoint_idx = 0
        self.reset_flag = False

        # Reset subscription
        self.reset_sub = self.node.create_subscription(
            String,
            "/env/reset",
            self.reset_callback,
            10
        )

    def initialise(self):
        self.node.get_logger().info("Initializing WaypointHandler...")
        try:
            self.path = self.blackboard.path
            self.path_length = len(self.path)
        except KeyError:
            self.node.get_logger().error("Path not found in blackboard")
            self.path = []
            self.path_length = 0
        self.waypoint_idx = 0

    def update(self):
        if self.reset_flag:
            self.node.get_logger().info("Resetting mission...")
            self.reset_flag = False
            self.waypoint_idx = 0
            return FAILURE

        if self.path is None or not self.path:
            self.node.get_logger().info("No path found.")
            return FAILURE

        if self.waypoint_idx >= self.path_length:
            self.node.get_logger().info("All waypoints processed.")
            return SUCCESS

        # Prepare the next waypoint
        next_waypoint = self.path[self.waypoint_idx]
        goal_position = json.dumps(next_waypoint)
        goal = MoveTo.Goal()
        goal.goal_position = goal_position
        self.blackboard.next_waypoint = goal

        self.node.get_logger().info(f"Waypoint {self.waypoint_idx + 1}/{self.path_length}: {next_waypoint}")
        self.waypoint_idx += 1
        return RUNNING

    def reset_callback(self, msg):
        self.reset_flag = True
        self.node.get_logger().info("Received mission reset signal.")


class WaypointMission(py_trees.composites.Sequence):
    def __init__(self, node):
        waypoint_handler = WaypointHandler(node)
        action_client = FromBlackboard(
            name="GoToWaypoint",
            action_type=MoveTo,
            action_name="/local_planner/move_to",
            key="next_waypoint",
            generate_feedback_message=self.feedback_message,
        )
        action_client.setup(node=node)

        super(WaypointMission, self).__init__(
            name="WaypointMission",
            memory=False,
            children=[waypoint_handler, action_client]
        )

    @staticmethod
    def feedback_message(msg):
        try:
            distance = json.loads(msg.feedback.feedback)['distance_to_goal']
            return f"{distance:.2f}%%"
        except (KeyError, json.JSONDecodeError):
            return "Invalid feedback format"
