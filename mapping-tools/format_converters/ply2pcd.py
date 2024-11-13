import open3d as o3d
import sys

def ply_to_pcd(input_ply, output_pcd):
    # Load the PLY file
    ply = o3d.io.read_point_cloud(input_ply)
    if not ply.has_points():
        sys.exit("Error: The point cloud is empty or could not be loaded.")
    
    # Save the point cloud as a PCD file
    o3d.io.write_point_cloud(output_pcd, ply)
    print(f"Conversion complete. PCD file saved to {output_pcd}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} input.ply output.pcd")
        sys.exit(1)

    input_ply = sys.argv[1]
    output_pcd = sys.argv[2]

    ply_to_pcd(input_ply, output_pcd)
