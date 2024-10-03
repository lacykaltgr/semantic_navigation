import open3d as o3d
import numpy as np

def remove_ceiling(pcd, height_threshold=1.0):
    points = np.asarray(pcd.points)
    filtered_points = points[points[:, 2] <= height_threshold]
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    return filtered_pcd


def simplify_mesh(mesh, factor, max_error):
    target_size = len(mesh.triangles) // factor
    mesh = mesh.simplify_quadric_decimation(target_size, maximum_error=max_error)
    return mesh

def poisson_reconstruction(input_pcd_file, output_ply_file, output_gltf_file, output_obj_file, output_simple_obj_file):
    # Load the point cloud from a PCD file
    pcd = o3d.io.read_point_cloud(input_pcd_file)

    print("Point cloud loaded successfully.")

    # Estimate normals for Poisson reconstruction
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    print("Normals estimated successfully.")
    pcd.orient_normals_consistent_tangent_plane(10)
    print("Normals oriented successfully.")
    

    # Perform Poisson reconstruction
    print("Performing Poisson reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=11)
    
    # Optionally, you can crop the mesh using a density filter to remove low-density triangles
    # This step is optional, depending on your needs
    print("Removing low-density vertices...")
    vertices_to_remove = densities < np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    print("Simplifying mesh...")
    simple_mesh = simplify_mesh(mesh, factor=10, max_error=10e-9)

    # Save the mesh as PLY
    print(f"Saving mesh to {output_ply_file}...")
    o3d.io.write_triangle_mesh(output_ply_file, mesh)

    # Save the mesh as GLTF
    # print(f"Saving mesh to {output_gltf_file}...")
    # o3d.io.write_triangle_mesh(output_gltf_file, mesh, write_ascii=False)

    # Save the mesh as OBJ
    print(f"Saving mesh to {output_obj_file}...")
    o3d.io.write_triangle_mesh(output_obj_file, mesh)


    # Save simplified mesh as OBJ
    print(f"Saving simplified mesh to {output_obj_file}...")
    o3d.io.write_triangle_mesh(output_simple_obj_file, simple_mesh)

    print("Mesh saved successfully.")

if __name__ == "__main__":
    input_pcd_file = "/workspace/ros2_ws/src/global_planner/resource/dense_18.pcd"
    output_ply_file = "/app/graph/mesh.ply"
    output_gltf_file = "/app/graph/mesh.gltf"
    output_obj_file = "/app/graph/mesh.obj"
    output_simple_obj_file = "/app/graph/simple_mesh.obj"

    poisson_reconstruction(input_pcd_file, output_ply_file, output_gltf_file, output_obj_file, output_simple_obj_file)