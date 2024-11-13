#!/bin/bash

SEMANTIC_MAPPING_DIR=/app

cd $SEMANTIC_MAPPING_DIR
git submodule update --recursive --remote --merge

# Install skeleton mapping
echo "Building Skeleton Finder"
if [ ! -d "$SEMANTIC_MAPPING_DIR/skeleton-mapping/build" ]; then
    mkdir $SEMANTIC_MAPPING_DIR/skeleton-mapping/build
fi
cd $SEMANTIC_MAPPING_DIR/skeleton-mapping/build
cmake .. -DCMAKE_BUILD_TYPE=Debug && make install
# TODO: add skeleton mapping to path


# Install concept fusion compact
conda run -n cf_compact pip install -e $SEMANTIC_MAPPING_DIR/conceptfusion-compact

# Install hworldmodel
# conda run -n isaaclab pip install -e $SEMANTIC_MAPPING_DIR/hworldmodel
# create symbolic link to /app/hworldmodel/ in /app/mission_ros_ws/src/env_manager/env_manager
# only if one does not exist
if [ ! -L "$SEMANTIC_MAPPING_DIR/mission_ros_ws/src/env_manager/env_manager/hworldmodel" ]; then
    ln -s $SEMANTIC_MAPPING_DIR/hworldmodel $SEMANTIC_MAPPING_DIR/mission_ros_ws/src/env_manager/env_manager/hworldmodel
fi
