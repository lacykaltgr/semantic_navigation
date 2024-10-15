import open3d as o3d
import argparse

def load_and_cut_colored_pointcloud(pcd_file_path, z_threshold, output_file_path):
    # Load the colored point cloud from the .pcd file
    pcd = o3d.io.read_point_cloud(pcd_file_path)
    if pcd.is_empty():
        print(f"Error: Failed to load point cloud from {pcd_file_path}")
        return

    # Filter points and colors where z <= z_threshold
    points = pcd.points
    colors = pcd.colors
    filtered_points = []
    filtered_colors = []

    for i in range(len(points)):
        if points[i][2] <= z_threshold:
            filtered_points.append(points[i])
            filtered_colors.append(colors[i])

    # Update point cloud with the filtered points and colors
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

    # Save the filtered point cloud to a new .pcd file
    o3d.io.write_point_cloud(output_file_path, pcd)
    print(f"Filtered colored point cloud saved to {output_file_path}")

def main():
    parser = argparse.ArgumentParser(description="Load and filter a colored point cloud based on z dimension.")
    parser.add_argument("pcd_file", help="Path to the input .pcd file")
    parser.add_argument("z_threshold", type=float, help="Z threshold for filtering")
    parser.add_argument("output_file", help="Path to save the filtered .pcd file")

    args = parser.parse_args()

    load_and_cut_colored_pointcloud(args.pcd_file, args.z_threshold, args.output_file)

if __name__ == "__main__":
    main()
