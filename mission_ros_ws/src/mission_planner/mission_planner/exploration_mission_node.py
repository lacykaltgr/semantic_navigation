import argparse
import rclpy
from rclpy.node import Node

from .missions.object_mission import QueryObject, FindRobot, FindPathToObject, WaypointMission, WhatIsTheQuery
from . import SUCCESS, WRITE
import py_trees
from std_msgs.msg import String

class MissionPlanner(Node):
    def __init__(self):
        super().__init__('mission_planner')

        self.query_sub = self.create_subscription(
            String,
            '/query',
            self.query_callback,
            10
        )
        self.setup_blackboard()

        self.blackboard = py_trees.blackboard.Client(name="Query")
        self.blackboard.register_key("query_lifo", access=WRITE)
        self.blackboard.query_lifo = []
        
        state_machine = self.createStateMachine()
        try:
            while True:
                state_machine.tick_once()
                rclpy.spin_once(self, timeout_sec=1.0)
            print("\n")
        except KeyboardInterrupt:
            pass

    def createStateMachine(self):
        mission = py_trees.composites.Sequence(
            name="ExplorationMission",
            memory=True,
            children=[
                FindNextBestView(self),
                FindRobot(self),
                FindPath(self),
                WaypointMission(self)
            ]
        )
        return mission
    
    def setup_blackboard(self):
        #py_trees.logging.level = py_trees.logging.Level.DEBUG
        py_trees.blackboard.Blackboard.enable_activity_stream(maximum_size=100)

    def query_callback(self, msg):
        query = msg.data
        self.blackboard.query_lifo.append(query)
        self.get_logger().info(f"Received query: {query}, added to query LIFO")


def main(args=None):
    rclpy.init(args=args)
    node = MissionPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
