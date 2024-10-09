#!/bin/bash

# This script is used to install the necessary dependencies for the semantic_navigation package

SEMANTIC_MAPPING_DIR=/app


# Install skeleton mapping
echo "Building Skeleton Finder"
mkdir $SEMANTIC_MAPPING_DIR/skeleton-mapping/build
cd $SEMANTIC_MAPPING_DIR/skeleton-mapping/build
cmake .. -DCMAKE_BUILD_TYPE=Debug && make
# TODO: add skeleton mapping to path

# Build isaac-ros ws
echo "Building Isaac ROS Humble Workspace"
source /opt/ros/humble/setup.bash && \
        cd /humble_ws && \
        colcon build
source /humble_ws/install/local_setup.bash #&& \
    #echo "source /humble_ws/install/local_setup.bash" >> ${HOME}/.bashrc


# Build ROS workspace
echo "Building DEMO ROS Workspace"
source /opt/ros/humble/setup.bash && \
     cd "$SEMANTIC_MAPPING_DIR/mission_ros_ws" && \
     colcon build
source "$SEMANTIC_MAPPING_DIR/install/setup.bash"

