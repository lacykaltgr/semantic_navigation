SEMANTIC_MAPPING_DIR=/Users/laszlofreund/code/ai/semantic_navigation
DOCKER_CURRENT_DIR=$SEMANTIC_MAPPING_DIR/docker/docker_pc
DATA_DIR=/Users/laszlofreund/data
DOCKER_CONTAINER_NAME=semantic_navigation_pc
MISSION_ROS_WS=$SEMANTIC_MAPPING_DIR/mission_ros_ws

mkdir $DOCKER_CURRENT_DIR/conda_environments
#cp ../conceptfusion-compact/environment.yaml /conda_environments/conceptfusion-compact.yaml
cp $MISSION_ROS_WS/src/cf_tools/environment.yaml $DOCKER_CURRENT_DIR/conda_environments/cf_tools.yaml
cp $MISSION_ROS_WS/src/local_planner/environment.yaml $DOCKER_CURRENT_DIR/conda_environments/local_planner.yaml
cp $MISSION_ROS_WS/src/mission_planner/environment.yaml $DOCKER_CURRENT_DIR/conda_environments/mission_planner.yaml
#cp $SEMANTIC_MAPPING_DIR/conceptfusion-compact/environment.yaml $DOCKER_CURRENT_DIR/conda_environments/cf_compact.yaml


eval docker build -t semantic_navigation_pc -f Dockerfile_macOS --platform=linux/amd64 --no-cache . 
eval docker run -dit \
  --name semantic_navigation_pc \
  -e DISPLAY="" \
  --network=host \
  -v ${SEMANTIC_MAPPING_DIR}:/app \
  -v ${DATA_DIR}:/workspace:rw \
  semantic_navigation_pc:latest
