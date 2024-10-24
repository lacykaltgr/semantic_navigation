import os
import json
import argparse
import numpy as np
import open3d as o3d

try:
    from utils.io import load_json, create_point_cloud, save_point_cloud, write_txt_lines
except ImportError:
    from .utils.io import load_json, create_point_cloud, save_point_cloud, write_txt_lines


def load_transformations(directories):
    """Load transformation translation vectors from JSON files in the specified directories."""
    translations = []
    image_paths = []

    for directory in directories:
        transform_dir = os.path.join(directory, 'tf')
        images_dir = os.path.join(directory, 'color')

        # Sort files by name to ensure correct order
        json_files = sorted([f for f in os.listdir(transform_dir) if f.endswith('.json')])
        color_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])

        assert len(json_files) == len(color_files), "Number of JSON files and images do not match"
        
        image_paths.extend([os.path.join(images_dir, f) for f in color_files])

        for json_file in json_files:
            file_path = os.path.join(transform_dir, json_file)
            data = load_json(file_path)
            translation = data['transform']['translation']
            position = np.array([translation['x'], translation['y'], translation['z']])
            translations.append(position)
    
    return translations, image_paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a PCD file from 3D pose transformations.")
    parser.add_argument('directories', nargs='+', help="List of input directories containing the transformation JSON files.")
    parser.add_argument('--output', required=True, help="Output PCD file path.")
    args = parser.parse_args()

    translations, paths = load_transformations(args.directories)
    point_cloud = create_point_cloud(translations)

    save_point_cloud(point_cloud, args.output)
    print(f"Point cloud saved to {args.output}")
    write_txt_lines(paths, args.output.replace('.pcd', '_paths.txt'))
    print(f"Image paths saved to {args.output.replace('.pcd', '_paths.txt')}")
    
