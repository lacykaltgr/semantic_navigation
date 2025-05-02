#!/bin/bash

NS_DATASET_PATH=$1
PLY_OUTPUT_DIR=$2
CONDA_CMD="conda run -n nerfstudio"

eval "${CONDA_CMD} ns-train splatfacto-big \
    --viewer.quit-on-train-completion=True \
    --pipeline.model.use_scale_regularization=True \
    --data ${NS_DATASET_PATH}"

ls ${PWD}
ls ${PWD}/outputs

# Find the latest training output
NS_DATASET_BASE_DIR=$(basename $NS_DATASET_PATH)
MODEL_OUTPUT_DIR="${PWD}/outputs/${NS_DATASET_BASE_DIR}/splatfacto/"
LATEST_TRAINING=$(ls -td $MODEL_OUTPUT_DIR/*/ | head -n 1)

# Export nerfstudio model
eval "${CONDA_CMD} ns-export gaussian-splat \
        --load-config ${LATEST_TRAINING}/config.yml \
        --output-dir ${PLY_OUTPUT_DIR}"