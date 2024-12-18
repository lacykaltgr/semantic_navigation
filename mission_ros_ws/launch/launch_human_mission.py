import os
from launch_node import LaunchConfig, LaunchNode, LaunchFile


GROQ_API_KEY = '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'
DATA_DIR = '/workspace/data_proc/skypark'


def generate_launch_description(data_dir):
    scene_description_file = os.path.join(data_dir, 'scene_description.json')
    #map_file = os.path.join(data_dir, 'merged_full_ds025_filtered_ds05_zt2_cut.pcd')
    nodes_file = os.path.join(data_dir, 'nodes.pcd')
    edges_file = os.path.join(data_dir, 'adjacency_matrix.txt')
    skeleton_config_file = os.path.join(data_dir, 'skeleton_config.yaml')


    return LaunchConfig([
        LaunchFile(
            name='rosbridge_server',
            package='rosbridge_server',
            launch_file='rosbridge_websocket_launch.xml'
        ),

        LaunchNode(
            name='rmq_bridge',
            package='satinav_bridge', 
            executable='rmq_bridge',
            conda_env='rmq_bridge_env',
        ),

        LaunchNode(
            name='satinav_bridge',
            package='satinav_bridge', 
            executable='satinav_bridge',
            conda_env='rmq_bridge_env',
        ),

        LaunchNode(
            name='cf_llm',
            package='cf_tools', 
            executable='cf_llm',
            conda_env='cf_tools_env',
            params={
                'scene_json_path': scene_description_file
            },
            env_variables={
                'GROQ_API_KEY': GROQ_API_KEY,
            }
        ),

        LaunchNode(
            name='global_planner',
            package='global_planner', 
            executable='global_planner_loaded',
            conda_env='base',
            params={
                'config_path': skeleton_config_file,
                #'map_path': map_file
                'nodes_path': nodes_file,
                'edges_path': edges_file
            }
        ),

        LaunchNode(
            name='mission_planner',
            package='mission_planner', 
            executable='human_mission',
            conda_env='mission_planner_env'
        )
    ])

if __name__ == '__main__':
    generate_launch_description(DATA_DIR).launch_all()