import os
from launch_node import LaunchConfig, LaunchNode


GROQ_API_KEY = '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'
DATA_DIR = '/workspace/data_proc/skypark'
CUDA_VISIBLE_DEVICES = '0,1,2'


def generate_launch_description(data_dir):
    scene_description_file = os.path.join(data_dir, 'scene_description.json')
    map_file = os.path.join(data_dir, 'merged_full_ds025_filtered_ds05.pcd')
    skeleton_config_file = os.path.join(data_dir, 'skeleton_config.yaml')


    return LaunchConfig([
        # ENV MANAGER
        LaunchNode(
            name='env_manager',
            package='env_manager', 
            executable='env_manager',
            conda_env='isaaclab',
            env_variables={
                'DISPLAY': "",
                'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES
            }
        ),

        # MOKK SLAM
        LaunchNode(
            name='mokk_slam',
            package='env_manager', 
            executable='mokk_slam',
            conda_env='isaaclab',
        ),

        # CF TOOLS
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

        # GLOBAL PLANNER
        LaunchNode(
            name='global_planner',
            package='global_planner', 
            executable='global_planner',
            conda_env='base',
            params={
                'map_path': map_file,
                'config_path': skeleton_config_file
            }
        ),

        # LOCAL PLANNER
        LaunchNode(
            name='local_planner',
            package='local_planner', 
            executable='local_planner',
            conda_env='local_planner_env'
        ),

        # MISSION PLANNER
        LaunchNode(
            name='mission_planner',
            package='mission_planner', 
            executable='object_mission',
            conda_env='mission_planner_env'
        )
    ])

if __name__ == '__main__':
    generate_launch_description(DATA_DIR).launch_all()