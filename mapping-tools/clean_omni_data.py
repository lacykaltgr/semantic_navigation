import os
import json
import argparse

def is_invalid_transform(transform):
    # Check if the translation is (0, 0, 0) and the rotation is (0, 0, 0, 1)
    translation = transform['transform']['translation']
    rotation = transform['transform']['rotation']
    
    return (translation['x'] == 0 and translation['y'] == 0 and translation['z'] == 0 and
            rotation['x'] == 0 and rotation['y'] == 0 and rotation['z'] == 0 and rotation['w'] == 1)

def remove_files(base_name, color_dir, depth_dir, tf_dir):
    # Remove color, depth, and transform files
    color_path = os.path.join(color_dir, f'{base_name}.png')
    depth_path = os.path.join(depth_dir, f'{base_name}.png')
    tf_path = os.path.join(tf_dir, f'{base_name}_tf.json')

    for file_path in [color_path, depth_path, tf_path]:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f'Removed: {file_path}')

def main(data_dir):
    # Paths to the data directories
    color_dir = os.path.join(data_dir, 'color')
    depth_dir = os.path.join(data_dir, 'depth')
    tf_dir = os.path.join(data_dir, 'tf')

    # Get sets of available base names for color, depth, and tf files
    color_files = {f.replace('.png', '') for f in os.listdir(color_dir) if f.endswith('.png')}
    depth_files = {f.replace('.png', '') for f in os.listdir(depth_dir) if f.endswith('.png')}
    tf_files = {f.replace('_tf.json', '') for f in os.listdir(tf_dir) if f.endswith('_tf.json')}

    # Find the common base names among all three sets
    common_files = color_files & depth_files & tf_files

    # Iterate over each common base name and check validity of the transformation
    for base_name in common_files:
        tf_path = os.path.join(tf_dir, f'{base_name}_tf.json')

        # Read the JSON file
        with open(tf_path, 'r') as file:
            transform_data = json.load(file)

        # Check if the transformation is invalid
        if is_invalid_transform(transform_data):
            remove_files(base_name, color_dir, depth_dir, tf_dir)

    # Find and remove any files that don't have corresponding color, depth, and tf files
    all_files = color_files | depth_files | tf_files
    for base_name in all_files:
        if base_name not in common_files:
            remove_files(base_name, color_dir, depth_dir, tf_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process RGB-D data and remove invalid files.')
    parser.add_argument('data_dir', type=str, help='Path to the data folder')
    args = parser.parse_args()

    main(args.data_dir)
