FROM nvcr.io/nvidia/isaac-sim:4.0.0

SHELL ["/bin/bash", "-c"]

ENV http_proxy=http://10.158.100.2:8080/
ENV https_proxy=http://10.158.100.2:8080/

ENV CONDA_DIR /opt/conda
RUN apt-get update && apt-get install -y wget
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_ROOT_USER_ACTION=ignore
ENV ISAACSIM_PATH=/isaac-sim/
ENV ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
ENV ISAACLAB_PATH=/IsaacLab/
ENV TERM=linux


# ROS2 Humble
RUN apt-get install -y --no-install-recommends \
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


# Download and extract LibTorch
RUN wget --quiet https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.4.1%2Bcu121.zip -O /libtorch.zip && \
    unzip /libtorch.zip -d / && \
    rm /libtorch.zip

# Set LibTorch environment variables
ENV CMAKE_PREFIX_PATH=/libtorch
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/libtorch/lib


# Copy the RMW specifications for ROS2
# https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_ros.html#enabling-the-ros-bridge-extension
COPY ./.ros/ ${DOCKER_USER_HOME}/.ros/

# clone isaac sim ros2 humble workspace
RUN git clone https://github.com/isaac-sim/IsaacSim-ros_workspaces.git && \
    cd IsaacSim-ros_workspaces && \
    cp -r humble_ws /humble_ws && \
    cd .. && rm -rf IsaacSim-ros_workspaces && \
    rm -rf /humble_ws/navigation
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    build-essential \
    python3-colcon-common-extensions
RUN cd /humble_ws && \
    rosdep init && rosdep update && rosdep install -i --from-path src --rosdistro humble -y

#RUN colcon build
# Default workspace is gonna be the isaac sim 
#RUN source /humble_ws/install/local_setup.bash && \
#    echo "source /humble_ws/install/local_setup.bash" >> ${HOME}/.bashrc

WORKDIR /

RUN /isaac-sim/kit/python/bin/python3 -m ensurepip --upgrade && \
    /isaac-sim/kit/python/bin/python3 -m pip install torch
RUN pip install --upgrade pip && pip install numpy==1.26.3 empy==3.3.4 lark
RUN mkdir conda_environments
#COPY ../conceptfusion-compact/environment.yaml /conda_environments/conceptfusion-compact.yaml
COPY ./conda_environments/cf_tools.yaml conda_environments
COPY ./conda_environments/local_planner.yaml conda_environments
COPY ./conda_environments/mission_planner.yaml conda_environments

# RUN conda env create -f /conda_environments/conceptfusion-compact.yaml
RUN conda env create -f /conda_environments/cf_tools.yaml
RUN conda env create -f /conda_environments/local_planner.yaml
RUN conda env create -f /conda_environments/mission_planner.yaml


# IsaacLab
RUN apt-get install cmake -y
RUN git clone https://github.com/isaac-sim/IsaacLab.git
RUN ln -sf /isaac-sim/  /IsaacLab/_isaac_sim
RUN /IsaacLab/isaaclab.sh --conda
SHELL ["conda", "run", "-n", "isaaclab", "/bin/bash", "-c"]
RUN ./IsaacLab/isaaclab.sh --install

RUN conda init

WORKDIR /app

RUN git config --global --add safe.directory /app
CMD chmod +x /app/init_script.sh && /app/init_script.sh