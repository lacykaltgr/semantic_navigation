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
    # Install rosdeps for extensions that declare a ros_ws in
    # their extension.toml
    #/IsaacLab/isaaclab.sh -p /IsaacLab/tools/install_deps.py rosdep IsaacLab/source/extensions && \
    apt -y autoremove && apt clean autoclean && \
    rm -rf /var/lib/apt/lists/* && \
    # Add sourcing of setup.bash to .bashrc
    echo "source /opt/ros/humble/setup.bash" >> ${HOME}/.bashrc
RUN source /opt/ros/humble/setup.bash


CMD chmod +x /app/init_script.sh && /app/init_script.sh