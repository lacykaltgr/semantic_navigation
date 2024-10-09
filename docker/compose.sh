

# turn this into a docker command 

SEMANTIC_MAPPING_DIR=/home/lfreund/projects/semantic_navigation
DOCKER_CURRENT_DIR=$SEMANTIC_MAPPING_DIR/docker
DATA_DIR=/mnt/data0/ros/documents
DOCKER_CONTAINER_NAME=semantic_navigation
DOCKER_CACHE_DIR=/home/lfreund/docker_cache

mkdir conda_environments
#cp ../conceptfusion-compact/environment.yaml /conda_environments/conceptfusion-compact.yaml
cp $SEMANTIC_MAPPING_DIR/mission_ros_ws/src/cf_tools/environment.yaml $DOCKER_CURRENT_DIR/conda_environments/cf_tools.yaml
cp $SEMANTIC_MAPPING_DIR/mission_ros_ws/src/local_planner/environment.yaml $DOCKER_CURRENT_DIR/conda_environments/local_planner.yaml
cp $SEMANTIC_MAPPING_DIR/mission_ros_ws/src/mission_planner/environment.yaml $DOCKER_CURRENT_DIR/conda_environments/mission_planner.yaml


docker build -t semantic_navigation:latest -f Dockerfile .
docker run -dit --gpus all \
  --name semantic_navigation \
  -e DISPLAY="" \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  --network=host \
  --runtime=nvidia \
  -v ${DOCKER_CACHE_DIR}/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
  -v ${DOCKER_CACHE_DIR}/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
  -v ${DOCKER_CACHE_DIR}/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
  -v ${DOCKER_CACHE_DIR}/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
  -v ${DOCKER_CACHE_DIR}/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
  -v ${DOCKER_CACHE_DIR}/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
  -v ${DOCKER_CACHE_DIR}/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
  -v ${DOCKER_CACHE_DIR}/docker/isaac-sim/.vscode-server:/root/.vscode-server:rw \
  -v ${SEMANTIC_MAPPING_DIR}:/app \
  -v ${DATA_DIR}:/workspace:rw \
  semantic_navigation:latest

  #-v /tmp/.X11-unix:/tmp/.X11-unix:ro \
