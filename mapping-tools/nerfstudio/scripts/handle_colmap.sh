#!/bin/bash

colmap_dir="/workspace"
output_dir="$colmap_dir/nerfstudio"

# TODO: add support for binary files
# check if required files exist, otherwise skip
if [ ! -f "$colmap_dir/cameras.txt" ] || \
    [ ! -f "$colmap_dir/images.txt" ] || \
    [ ! -f "$colmap_dir/points3D.txt" ]; then
    echo "Skipping $colmap_dir, missing required files."
    # exit the program
    exit 1
fi

# Convert colmap data to nerfstudio data
nerfstudio_dataset_path="$output_dir/nerfstudio_dataset"
conda run -n nerfstudio python /app/scripts/colmap2nerfstudio.py \
    --input_model_dir "$colmap_dir" \
    --input_images_dir "$colmap_dir/images" \
    --output_dir "$nerfstudio_dataset_path"

chmod +x /app/scripts/build_splat_docker.sh
eval ./app/scripts/build_splat_docker.sh "$nerfstudio_dataset_path" "$output_dir"