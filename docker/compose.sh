

# turn this into a docker command 

SEMANTIC_MAPPING_DIR=.
DATA_DIR=/mnt/data0/ros/documents
DOCKER_CONTAINER_NAME=semantic_navigation


docker build -t semantic_navigation:latest -f Dockerfile .
docker run -dit --rm --gpus all \
  --name ${DOCKER_CONTAINER_NAME} \
  -e DISPLAY=$DISPLAY \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  --network=host
  --runtime=nvidia

  -v ${SEMANTIC_MAPPING_DIR}/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
  -v ${SEMANTIC_MAPPING_DIR}/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
  -v ${SEMANTIC_MAPPING_DIR}/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
  -v ${SEMANTIC_MAPPING_DIR}/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
  -v ${SEMANTIC_MAPPING_DIR}/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
  -v ${SEMANTIC_MAPPING_DIR}/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
  -v ${SEMANTIC_MAPPING_DIR}/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
  -v ${SEMANTIC_MAPPING_DIR}/docker/isaac-sim/.vscode-server:/root/.vscode-server:rw \
  -v ${DATA_DIR}:/workspace:rw \
  #-v /tmp/.X11-unix:/tmp/.X11-unix:ro \

  semantic_navigation:latest
