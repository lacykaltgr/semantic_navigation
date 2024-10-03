# This tool takes an existing colmap model (images, cameras, points3D) plus the corresponding images folder as input,
# and it converts to a format that NerfStudio can load. The images get copied and automatically resized.
# The image poses are converted to NerfStudio's coordinate system and together with camera intrinsics they are written into a transforms.json file.
# See https://docs.nerf.studio/en/latest/quickstart/custom_dataset.html

# Author: Gabor Soros (gabor.soros@nokia-bell-labs.com)
# 15.12.2022

# Example usage:
#
# conda activate nerfstudio
#
# python convert_colmap_model_to_nerfstudio.py \
#  --input_model_dir "$HOME/data/221212NerfTestChessK4A-1/cvsl_colmap" \
#  --input_images_dir "$HOME/data/221212NerfTestChessK4A-1/cvsl_colmap/images" \
#  --output_dir "$HOME/data/221212NerfTestChessK4A-1/nerfstudio_dataset"
#
# ns-train nerfacto --data $HOME/data/221212NerfTestChessK4A-1/nerfstudio_dataset


import argparse
import shutil
from pathlib import Path

from nerfstudio.process_data import colmap_utils, process_data_utils
#from nerfstudio.process_data.process_data_utils import CAMERA_MODELS # not needed since NerfStudio 2023.02.26
from nerfstudio.data.utils import colmap_parsing_utils # added in NerfStudio 2023.02.26

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Colmap to NerfStudio converter",
        description="This script converts an existing colmap model and images into NerfStudio's input format",
    )
    parser.add_argument("--input_model_dir", type=Path, required=True)
    parser.add_argument("--input_images_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    print("Input colmap model dir: " + str(args.input_model_dir))
    print("Input images dir: " + str(args.input_images_dir))
    print("Output NerfStudio dataset dir: " + str(args.output_dir))

    camera_type: str = "perspective"  # ["perspective", "fisheye"]
    num_downscales: int = 3

    # Detect colmap model type and if TXT, convert it to BIN
    if colmap_parsing_utils.detect_model_format(args.input_model_dir, ".txt"):
        print("Converting colmap model from TXT to BIN format")
        cameras = colmap_parsing_utils.read_cameras_text(args.input_model_dir / "cameras.txt")
        images = colmap_parsing_utils.read_images_text(args.input_model_dir / "images.txt")
        points3D = colmap_parsing_utils.read_points3D_text(args.input_model_dir / "points3D.txt")
        colmap_parsing_utils.write_cameras_binary(cameras, args.input_model_dir / "cameras.bin")
        colmap_parsing_utils.write_images_binary(images, args.input_model_dir / "images.bin")
        colmap_parsing_utils.write_points3D_binary(points3D, args.input_model_dir / "points3D.bin")

    # Check whether conversion was successful
    if not colmap_parsing_utils.detect_model_format(args.input_model_dir, ".bin"):
        print("ERROR: input_model is not in binary format")
        exit(-1)

    # create output directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    # Copy images to output directory
    # Note that the copies will be named like frame_xxxxx
    num_frames = process_data_utils.copy_images(args.input_images_dir, image_dir=image_dir, verbose=args.verbose)
    print(f"Starting with {num_frames} images")

    # Downscale images (also named in frame_xxxxx format)
    status = process_data_utils.downscale_images(image_dir, num_downscales, verbose=args.verbose)
    print(status)

    # write transforms.json based on the colmap poses and camera model
    num_matched_frames = colmap_utils.colmap_to_json(
        recon_dir=args.input_model_dir, # needed since NerfStudio Feb 2023
        output_dir=args.output_dir
        #cameras_path=args.input_model_dir / "cameras.bin",
        #images_path=args.input_model_dir / "images.bin",
        #camera_model=CAMERA_MODELS[camera_type], # not needed since NerfStudio 2023.02.26
    )
    print(str(num_matched_frames) + " poses converted to transforms.json")

    # WARNING: the transforms.json produced by this step will have the original image filenames from images.bin, and not the frame_xxxxx format.
    # This is a problem because the converted dataset won't have the original filenames anymore. A quick and dirty workaround is that we copy
    # all original images into the output images folder, so every image will be twice in that folder (once with its original name and once as frame_xxxxx)
    # Note that any image manipulations such as crop_border_pixels are not allowed with this workaround.
    # TODO: colmap_to_json has a new parameter `image_rename_map` which could fix this issue.
    allowed_exts = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
    input_image_paths = sorted([p for p in args.input_images_dir.glob("[!.]*") if p.suffix.lower() in allowed_exts])
    for idx, input_image_path in enumerate(input_image_paths):
        if args.verbose:
            print(f"Copying image {idx + 1} of {len(input_image_paths)}...")
        copied_image_path = image_dir / input_image_path.name
        shutil.copy(input_image_path, copied_image_path)


    print("ALL DONE.")
    exit(0)
