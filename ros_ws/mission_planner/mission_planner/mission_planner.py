import argparse
import json
from rclpy.node import Node
import rclpy
import py_trees

from .missions.waypoint_mission import WaypointMission


class MissionPlanner(Node):
    def __init__(self, mission_file_path):
        super().__init__('mission_planner')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('mission_file_path', mission_file_path)
            ]
        )
        self.yaml_file_path = self.get_parameter("mission_file_path").value
        self.missions_data = self.readMissionsData(self.yaml_file_path)

        self.state_machine = self.createStateMachine(self.missions_data)
        self.setup_blackboard()
        self.state_machine.tick_once()

    def createStateMachine(self, mission_data):
        root = py_trees.composites.Sequence(
            name="WaypointMissionPlanner",
            memory=False,
            children=[
                WaypointMission(node=self, mission_data=mission_data)
            ]
        )
        return root
    
    def readMissionsData(self, yaml_file_path):
        with open(yaml_file_path, 'r') as file:
            missions_data = json.load(file)
        return missions_data
    
    def setup_blackboard(self):
        py_trees.logging.level = py_trees.logging.Level.DEBUG
        py_trees.blackboard.Blackboard.enable_activity_stream(maximum_size=100)
        

def main(args=None):
    rclpy.init(args=args)
    node = MissionPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()