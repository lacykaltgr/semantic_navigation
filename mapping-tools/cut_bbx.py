import open3d as o3d
import numpy as np
import argparse
import sys

# -11 -11.3
# 57.6 58.4
# 0.2 2

def parse_arguments():
    parser = argparse.ArgumentParser(description="Remove points within a specified bounding box from a PCD file.")
    parser.add_argument("input_pcd", type=str, help="Path to the input PCD file.")
    parser.add_argument("output_pcd", type=str, help="Path to save the corrected PCD file.")
    parser.add_argument("xmin", type=float, help="Minimum x-coordinate of the bounding box.")
    parser.add_argument("xmax", type=float, help="Minimum y-coordinate of the bounding box.")
    parser.add_argument("ymin", type=float, help="Minimum z-coordinate of the bounding box.")
    parser.add_argument("ymax", type=float, help="Maximum x-coordinate of the bounding box.")
    parser.add_argument("zmin", type=float, help="Maximum y-coordinate of the bounding box.")
    parser.add_argument("zmax", type=float, help="Maximum z-coordinate of the bounding box.")
    return parser.parse_args()

def remove_points_within_bounding_box(pcd, bbox_min, bbox_max):
    # Create a boolean mask for points outside the bounding box
    points = np.asarray(pcd.points)
    mask = ~((points[:, 0] >= bbox_min[0]) & (points[:, 0] <= bbox_max[0]) &
             (points[:, 1] >= bbox_min[1]) & (points[:, 1] <= bbox_max[1]) &
             (points[:, 2] >= bbox_min[2]) & (points[:, 2] <= bbox_max[2]))

    # Filter the points based on the mask
    pcd.points = o3d.utility.Vector3dVector(points[mask])

    # If colors are available, filter them as well
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        pcd.colors = o3d.utility.Vector3dVector(colors[mask])

    # If normals are available, filter them as well
    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
        pcd.normals = o3d.utility.Vector3dVector(normals[mask])

    return pcd

def main():
    args = parse_arguments()

    # Load the point cloud
    pcd = o3d.io.read_point_cloud(args.input_pcd)
    if not pcd.has_points():
        sys.exit("Error: The point cloud is empty or could not be loaded.")

    # Bounding box coordinates
    bbox_min = np.array([args.xmin, args.ymin, args.zmin])
    bbox_max = np.array([args.xmax, args.ymax, args.zmax])

    # Remove points within the bounding box
    pcd = remove_points_within_bounding_box(pcd, bbox_min, bbox_max)

    # Save the corrected point cloud
    o3d.io.write_point_cloud(args.output_pcd, pcd)
    print(f"Corrected PCD saved to {args.output_pcd}")

if __name__ == "__main__":
    main()
