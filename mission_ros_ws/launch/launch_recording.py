import os
from launch_node import LaunchConfig, LaunchNode


DATA_DIR = '/workspace/data/data15'
CUDA_VISIBLE_DEVICES = '0,1,2'


def generate_launch_description(data_dir):
    return LaunchConfig([
        # DATA SAVER
        LaunchNode(
            name='data_saver',
            package='data_saver', 
            executable='data_saver_node',
            conda_env='base',
            params={
                "save_path": DATA_DIR
            }
        ),

        # TELEOP
        LaunchNode(
            name='teleop',
            package='teleop_twist_keyboard', 
            executable='teleop_twist_keyboard',
            conda_env='base',
        ),
    ])

if __name__ == '__main__':
    generate_launch_description(DATA_DIR).launch_all()