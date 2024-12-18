import os

from launch import LaunchDescription
from launch_ros.actions import Node, IncludeLaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import AnyLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    ws_launch_file = os.path.join(
        get_package_share_directory('rosbridge_server'),
        'launch',
        'rosbridge_websocket_launch.xml'
    )

    data_dir_arg = DeclareLaunchArgument(
        'data_dir',
        default_value='/workspace/data_proc/skypark'
    )

    return LaunchDescription([
        data_dir_arg,
        
        # ros bridge websocket
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(ws_launch_file)
        ),

        # rmq_bridge
        Node(
            package='satinav_bridge',
            executable='rmq_bridge',
            name='rmq_bridge'
        ),

        # satinav_bridge
        Node(
            package='satinav_bridge',
            executable='satinav_bridge',
            name='satinav_bridge'
        ),

        # cf_llm
        Node(
            package='cf_tools',
            executable='cf_llm',
            name='cf_llm'
        ),

        # global_planner
        Node(
            package='global_planner',
            executable='global_planner',
            name='global_planner'
        ),

        # mission_planner
        Node(
            package='mission_planner',
            executable='human_mission',
            name='mission_planner'
        ),
    ])