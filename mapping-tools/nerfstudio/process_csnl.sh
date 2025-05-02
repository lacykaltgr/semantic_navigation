#!/bin/bash

# $1: the path to raw data folder
# $2: path to root folder


# Step 1: Find all directories in the raw data folder
for dir in $1/*; do

    # get the name of the directory
    dir_name=$(basename $dir)

    # check if camera, images, and points3D.bin exist
    # if not, continue to the next directory
    if [ ! -f "$dir/cvsl_colmap_sequence/cameras.txt" ] || [ ! -f "$dir/cvsl_colmap_sequence/images.txt" ] || [ ! -f "$dir/cvsl_colmap_sequence/points3D.txt" ]; then
        echo "Skipping $dir_name"
        continue
    fi

    # Step 2: Process each directory
    # Find cvsl_colmap_sequence inside each directory
    # Copy the cvsl_colmap_sequence to colmap_datasets under name: $dir-colmap
    mkdir -p $2/colmap_datasets/$dir_name-colmap
    cp -r "$dir/cvsl_colmap_sequence/"* $2/colmap_datasets/$dir_name-colmap
    # remove points3D.txt, as it is empty in _sequence datasets
    rm "$2/colmap_datasets/$dir_name-colmap/points3D.txt"
    # add points3D.txt from the original dataset
    cp "$dir/cvsl_colmap/points3D.txt" $2/colmap_datasets/$dir_name-colmap
    # remove all txt files in $2/colmap_datasets/$dir_name-colmap
    # rm "$2/colmap_datasets/$dir_name-colmap/"*.txt

    # Step 3: turn colmap data to nerfstudio data
    python scripts/colmap2nerfstudio.py \
        --input_model_dir "$2/colmap_datasets/$dir_name-colmap" \
        --input_images_dir "$2/colmap_datasets/$dir_name-colmap/images" \
        --output_dir "$2/nerfstudio_datasets/$dir_name-nerfstudio" \

    # Step 4: run nerfstudio training
    ns-train splatfacto-big \
        --viewer.quit-on-train-completion=True \
        --pipeline.model.use_scale_regularization=True \
        --data "$2/nerfstudio_datasets/$dir_name-nerfstudio"

    # find the latest training output
    OUTPUT_DIR="$2/outputs/$dir_name-nerfstudio/splatfacto/"
    LATEST_TRAINING=$(ls -td $OUTPUT_DIR/*/ | head -n 1)

    # Step 5: export nerfstudio model
    ns-export gaussian-splat \
        --load-config $LATEST_TRAINING/config.yml \
        --output-dir $LATEST_TRAINING

done