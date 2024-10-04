from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mission_planner',
            executable='object_mission',
            name='mission_planner',
            ros_arguments=[
                'query:=sofa',
            ]
        ),
        Node(
            package='cf_tools',
            executable='cf_tools',
            name='cf_tools',
            ros_arguments=[
                'query_model:=clip',
                'result_path:=/root/ros2_ws/src/cf_tools/resource/pcd_mapping.pkl.gz',
            ]
        ),
        Node(
            package='global_planner',
            executable='global_planner',
            name='global_planner',
            ros_arguments=[
                'query:=sofa',
            ]
        )
    ])