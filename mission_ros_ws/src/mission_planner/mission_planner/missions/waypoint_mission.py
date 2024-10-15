from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
import py_trees
from mission_planner_interfaces.action import MoveTo
from . import READ, SUCCESS, FAILURE, WRITE, RUNNING
from action_msgs.msg import GoalStatus
from py_trees_ros.action_clients import FromBlackboard
import json


class NextWaypoint(py_trees.behaviour.Behaviour):
    def __init__(self, node):
        super(NextWaypoint, self).__init__(name="NextWaypoint")
        self.node = node

        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key="path", access=READ)
        self.blackboard.register_key(key="next_waypoint", access=WRITE)

        self.path = None
        self.waypoint_idx = 0

    def initialise(self):
        self.node.get_logger().info("Getting next waypoint from blackboard")
        if self.path is None:
            try:
                self.path = self.blackboard.path
            except KeyError:
                self.node.get_logger().error("Path not found in blackboard")


    def update(self):
        # write next waypoint to bb
        if self.path is None:
            self.node.get_logger().info("Path not found in blackboard")
            return FAILURE
        
        next_waypoint = self.path[self.waypoint_idx]
        goal_position = json.dumps(next_waypoint)
        goal = MoveTo.Goal()
        goal.goal_position = goal_position
        self.blackboard.next_waypoint = goal
        self.node.get_logger().info(f"Next waypoint: {next_waypoint}")
        self.waypoint_idx += 1
        if self.waypoint_idx >= len(self.path):
            self.waypoint_idx = 0
        return SUCCESS


class HasMoreWaypoints(py_trees.behaviour.Behaviour):
    def __init__(self, node):
        super(HasMoreWaypoints, self).__init__(name="HasMoreWaypoints")
        self.node = node

        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key="path", access=READ)

        self.path_length = None
        self.waypoint_idx = 0

    def initialise(self):
        self.waypoint_idx = 0
        if self.path_length is None:
            try:
                self.path_length = len(self.blackboard.path)
            except KeyError:
                self.node.get_logger().error("Path not found in blackboard")


    def update(self):
        # write next waypoint to bb
        if self.path_length is None:
            self.node.get_logger().info("Path not found in blackboard")
            return FAILURE
        
        self.node.get_logger().info(f"Checking if there are more waypoints: {self.waypoint_idx}/{self.path_length}")
        self.waypoint_idx += 1
        if self.waypoint_idx < self.path_length:
            return RUNNING
        else:
            return SUCCESS


class WaypointMission(py_trees.composites.Sequence):
    def __init__(self, node):

        next_waypoint_setter = NextWaypoint(node)
        action_client = FromBlackboard(
            name="GoToWaypoint",
            action_type=MoveTo,
            action_name="/local_planner/move_to",
            key="next_waypoint",
            generate_feedback_message=lambda msg: f"{json.loads(msg.feedback.feedback)['distance_to_goal']:.2f}%%",
        )
        action_client.setup(node=node)

        goto_waypoint_seq = py_trees.composites.Sequence(
            name="GoToWaypointSequence",
            children=[
                next_waypoint_setter,
                action_client,
            ],
            memory=True
        )

        super(WaypointMission, self).__init__(
            name="WaypointMission",
            memory=False,
            children=[
                goto_waypoint_seq,
                HasMoreWaypoints(node=node),
            ]
        )