#!/bin/sh

# This script is used to install the necessary dependencies for the semantic_navigation package

SEMANTIC_MAPPING_DIR = /app


# Install skeleton mapping
mkdir $SEMANTIC_MAPPING_DIR/skeleton_mapping/build
cd $SEMANTIC_MAPPING_DIR/skeleton_mapping/build
cmake .. -DCMAKE_BUILD_TYPE=Debug && make install
# TODO: add skeleton mapping to path

# Install concept fusion compact
cd $SEMANTIC_MAPPING_DIR/conceptfusion-compact
conda env create -f environment.yml

# Install hworldmodel
cd $SEMANTIC_MAPPING_DIR/hworldmodel
conda env create -f environment.yml

# Install ros node envs
cd $SEMANTIC_MAPPING_DIR/ros_ws
conda env create -f ./src/cf_tools/environment.yml
conda env create -f ./src/local_planner/environment.yml
conda env create -f ./src/mission_planner/environment.yml
# env_manager uses conda env from hworldmodel
#conda env create -f ./env_manager/environment.yml

# Build ROS workspace
colcon build --symlink-install
source install/setup.bash

