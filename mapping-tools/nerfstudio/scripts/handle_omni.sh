#!/bin/bash

omni_dir=$1
output_dir=$2

# TODO: add support for binary files
# check if required files exist, otherwise skip
if [ ! -f "$colmap_dir/cameras.txt" ] || \
    [ ! -f "$colmap_dir/images.txt" ] || \
    [ ! -f "$colmap_dir/points3D.txt" ]; then
    echo "Skipping $colmap_dir, missing required files."
    # exit the program
    exit 1
fi

# Convert omni data to colmap data
python3 scripts/omni2colmap.py \
    --input_omni_dir "$omni_dir" \
    --output_colmap_dir "$output_dir/colmap_dataset" \
    --ext ".txt"

# Convert colmap data to nerfstudio data
python3 scripts/colmap2nerfstudio.py \
    --input_model_dir "$output_dir/colmap_dataset" \
    --input_images_dir "$output_dir/colmap_dataset/images" \
    --output_dir "$output_dir/nerfstudio_dataset"