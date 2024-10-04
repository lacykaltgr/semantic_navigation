FROM nvcr.io/nvidia/pytorch:23.10-py3

ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

ARG DEBIAN_FRONTEND=noninteractive

# ROS2 Humble
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    # Install ROS2 Humble \
    software-properties-common && \
    add-apt-repository universe && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo jammy) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-desktop \
    ros-humble-vision-msgs \
    # Install both FastRTPS and CycloneDDS
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-rmw-fastrtps-cpp \
    # This includes various dev tools including colcon
    ros-dev-tools && \
    apt -y autoremove && apt clean autoclean && \
    rm -rf /var/lib/apt/lists/* && \
    # Add sourcing of setup.bash to .bashrc
    echo "source /opt/ros/humble/setup.bash" >> ${HOME}/.bashrc
RUN source /opt/ros/humble/setup.bash


# Copy the RMW specifications for ROS2
# https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_ros.html#enabling-the-ros-bridge-extension
COPY ./.ros/ ${DOCKER_USER_HOME}/.ros/

# clone isaac sim ros2 humble workspace
RUN git clone https://github.com/isaac-sim/IsaacSim-ros_workspaces.git && \
    cd IsaacSim-ros_workspaces && \
    cp -r ./humble_ws /app && \
    cd .. && rm -rf IsaacSim-ros_workspaces
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    build-essential \
    python3-colcon-common-extensions
RUN cd /app/humble_ws && \
    rosdep init && rosdep update && rosdep install -i --from-path src --rosdistro humble -y
RUN colcon build
# Default workspace is gonna be the isaac sim workspace
RUN source /app/humble_ws/install/local_setup.bash && \
    echo "source /app/humble_ws/install/local_setup.bash" >> ${HOME}/.bashrc


CMD chmod +x /app/init_script.sh && /app/init_script.sh