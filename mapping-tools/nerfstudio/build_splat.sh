#!/bin/bash

# implemented:
# - nerfstudio: path to dataset
# - colmap: path to colmap dataset

# check for the correct number of arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <path_to_dataset> <dataset_type>"
    echo "Dataset types: omni, colmap, nerfstudio, sequence"
    exit 1
fi

# assign input arguments to variables
dataset_path=$1
dataset_type=$2 # TODO: detect automatically

# create output dir at dataset_dir/nerfstudio
nerfstudio_scripts_dir="/home/lfreund/projects/semantic_navigation/mapping-tools/nerfstudio"

# Define constants
DOCKER_IMAGE="nerfstudio:latest"
CACHE_DIR="$HOME/.cache"
SHM_SIZE="12gb"
DOCKER_CMD="docker run --gpus all  \
            -v ${nerfstudio_scripts_dir}:/app/ \
            -v ${dataset_path}:/workspace/  \
            -v ${CACHE_DIR}:/home/user/.cache/  \
            -p 7007:7007 --rm -it -q --shm-size=${SHM_SIZE} ${DOCKER_IMAGE}"

# Process the dataset based on its type
case "$dataset_type" in
    nerfstudio)
        echo "Nerstudio Dataset: no further processing needed..."
        COMMAND="${DOCKER_CMD} ./app/scripts/build_splat_docker.sh /workspace /workspace/nerfstudio"
        ;;
    colmap)
        echo "Colmap Dataset, processing..."
        COMMAND="${DOCKER_CMD} ./app/scripts/handle_colmap.sh"
        ;;
    sequence)
        echo "Processing sequence dataset..."
        echo "extends DOCKER command with calling ns-process-data"
        echo "Not implemented yet."
        exit 1
        ;;
    omni)
        echo "Omniverse Dataset, processing..."
        COMMAND="${DOCKER_CMD} ./app/scripts/handle_omni.sh"
        ;;
    *)
        echo "Unsupported dataset type: $dataset_type"
        echo "Supported types: omni, colmap, nerfstudio, sequence"
        exit 1
        ;;
esac

echo "Starting docker container..."
echo "Running command: $COMMAND"
eval $COMMAND
echo "Docker container finished."
echo "Gaussian splat creation completed."
