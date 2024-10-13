#!/bin/bash

# This script is used to install the necessary dependencies for the semantic_navigation package

SEMANTIC_MAPPING_DIR=/app

# activate base conda environment
conda deactivate
# install catkin_pkg with pip if not installed
if ! python -c "import catkin_pkg" &> /dev/null; then
    pip install catkin_pkg
fi

source "$SEMANTIC_MAPPING_DIR/docker/update_submodules.sh"

# Build isaac-ros ws
#echo "Building Isaac ROS Humble Workspace"
#source /opt/ros/humble/setup.bash && \
#        cd /humble_ws && \
#        colcon build
#source /humble_ws/install/local_setup.bash #&& \
    #echo "source /humble_ws/install/local_setup.bash" >> ${HOME}/.bashrc


# Build ROS workspace
echo "Building DEMO ROS Workspace"
source /opt/ros/humble/setup.bash && \
     cd "$SEMANTIC_MAPPING_DIR/mission_ros_ws" && \
     colcon build
source "$SEMANTIC_MAPPING_DIR/mission_ros_ws/install/setup.bash"

cd $SEMANTIC_MAPPING_DIR

