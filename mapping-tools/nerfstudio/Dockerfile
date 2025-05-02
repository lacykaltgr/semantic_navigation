FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]

ENV http_proxy=http://10.158.100.2:8080/
ENV https_proxy=http://10.158.100.2:8080/

ENV CONDA_DIR /opt/conda
RUN apt-get update && apt-get install -y wget
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ENV DEBIAN_FRONTEND=noninteractive

COPY ./environment.yaml /tmp/environment.yaml

RUN apt-get install -y git
RUN conda env create -f /tmp/environment.yaml
ENV TCNN_CUDA_ARCHITECTURES=86
RUN conda run -n nerfstudio pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN conda run -n nerfstudio ns-install-cli

# raises error if not done
RUN conda run -n nerfstudio pip uninstall gsplat -y
ENV TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6
RUN conda run -n nerfstudio pip install git+https://github.com/nerfstudio-project/gsplat.git