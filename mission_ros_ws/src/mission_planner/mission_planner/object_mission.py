import argparse
import rclpy
from rclpy.node import Node

from .missions.object_mission import ObjectMission
from . import SUCCESS
import py_trees

class MissionPlanner(Node):
    def __init__(self, query="sofa"):
        super().__init__('mission_planner')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('query', query)
            ]
        )
        self.query = self.get_parameter("query").value

        self.setup_blackboard()
        
        state_machine = self.createStateMachine()
        try:
            while True:
                state_machine.tick_once()
                rclpy.spin_once(self)
            print("\n")
        except KeyboardInterrupt:
            pass

        
        

    def createStateMachine(self):
        mission = ObjectMission(
            node=self, 
            query=self.query
        )
        return mission
    
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
