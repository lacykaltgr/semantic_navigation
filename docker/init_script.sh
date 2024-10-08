#!/bin/sh

# This script is used to install the necessary dependencies for the semantic_navigation package

SEMANTIC_MAPPING_DIR = /app


# Install skeleton mapping
mkdir $SEMANTIC_MAPPING_DIR/skeleton_mapping/build
cd $SEMANTIC_MAPPING_DIR/skeleton_mapping/build
cmake .. -DCMAKE_BUILD_TYPE=Debug && make install
# TODO: add skeleton mapping to path

# Install concept fusion compact

# env_manager uses conda env from hworldmodel
#conda env create -f ./env_manager/environment.yml

# Build ROS workspace
colcon build --symlink-install
source install/setup.bash

