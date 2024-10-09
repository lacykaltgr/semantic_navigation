#!/bin/bash

SEMANTIC_MAPPING_DIR=/app

cd $SEMANTIC_MAPPING_DIR
git submodule update --recursive --remote --merge

# Install skeleton mapping
echo "Building Skeleton Finder"
mkdir $SEMANTIC_MAPPING_DIR/skeleton-mapping/build
cd $SEMANTIC_MAPPING_DIR/skeleton-mapping/build
cmake .. -DCMAKE_BUILD_TYPE=Debug && make
# TODO: add skeleton mapping to path


# Install concept fusion compact
conda run -n cf_compact your_command

# Install hworldmodel
conda run -n isaaclab your_command

