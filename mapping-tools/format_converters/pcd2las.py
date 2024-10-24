import open3d as o3d
import laspy
import numpy as np
import sys

def pcd_to_las(input_pcd, output_las):
    # Load the PCD file
    pcd = o3d.io.read_point_cloud(input_pcd)
    if not pcd.has_points():
        sys.exit("Error: The point cloud is empty or could not be loaded.")
    
    # Extract points
    points = np.asarray(pcd.points)
    
    # Extract colors if available
    if pcd.has_colors():
        colors = np.asarray(pcd.colors) * 255
        colors = colors.astype(np.uint16)  # Scale colors to 16-bit integers
    else:
        # Use default color (255, 255, 255) if colors are not available
        colors = np.full((points.shape[0], 3), 255, dtype=np.uint16)
    
    # Create a new LAS file
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = np.min(points, axis=0)
    header.scales = np.array([0.01, 0.01, 0.01])  # Adjust scaling as needed

    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    las.red = colors[:, 0]
    las.green = colors[:, 1]
    las.blue = colors[:, 2]

    # Write the LAS file
    las.write(output_las)
    print(f"Conversion complete. LAS file saved to {output_las}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} input.pcd output.las")
        sys.exit(1)

    input_pcd = sys.argv[1]
    output_las = sys.argv[2]

    pcd_to_las(input_pcd, output_las)
