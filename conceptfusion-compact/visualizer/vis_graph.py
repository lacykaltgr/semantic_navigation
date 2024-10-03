import open3d as o3d
import numpy as np



# Load the point cloud
def load_point_cloud_from_pcl(file_path: str) -> o3d.geometry.PointCloud:
    # Load the PCL format into Open3D's PointCloud
    pcd = o3d.io.read_point_cloud(file_path)
    print(pcd)
    return pcd

# Load connections from connections.txt
def load_edges_from_file(file_path: str):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    edges = []
    print(len(lines))
    for i in range(0, len(lines), 3):
        # Parse each pair of lines as two 3D points
        point1 = np.array([float(x) for x in lines[i].split()])
        point2 = np.array([float(x) for x in lines[i+1].split()])
        edges.append((point1, point2))

    return edges

# Add edges as lines for visualization
def create_graph_edges(edges, offst=False):
    points = []
    lines = []

    for i, (point1, point2) in enumerate(edges):
        points.append(point1)
        points.append(point2)
        lines.append([2 * i, 2 * i + 1])
        
        if point1.shape[0] != 3 or point2.shape[0] != 3:
            print(point1, point2) 

    print("creating lineset")


    points = np.array(points)
    lines = np.array(lines)

    points = o3d.utility.Vector3dVector(points)
    print("points ok")
    lines = o3d.utility.Vector2iVector(lines)
    print("lines ok")

    # Create line set for visualization
    line_set = o3d.geometry.LineSet(
         points, lines
    )

    colors = [[1, 0, 0] for _ in range(len(lines))]  # Red color for each line
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

def visualize_graph(point_cloud_file: str, node_cloud_file: str, connections_file: str):
    # Load point cloud
    pcd = load_point_cloud_from_pcl(node_cloud_file)
    print("Loaded point cloud")

    # Load edges
    edges = load_edges_from_file(connections_file)
    print("Loaded edges")

    # Create graph lines
    graph_lines = create_graph_edges(edges)
    print("Created edges")

    pcd_house = load_point_cloud_from_pcl(point_cloud_file)
    vis = o3d.visualization.VisualizerWithKeyCallback()

    # Visualize the point cloud and graph lines together
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(graph_lines)

    visualize_graph.point_cloud_visible = False
    def toggle_point_cloud(vis):
        if not visualize_graph.point_cloud_visible:
            vis.add_geometry(pcd_house, reset_bounding_box=False)
            print("adding geom")
        else:
            vis.remove_geometry(pcd_house, reset_bounding_box=False)
            print("removing geom")
        visualize_graph.point_cloud_visible = not visualize_graph.point_cloud_visible
        return False  # Stops the event propagation

    vis.register_key_callback(32, toggle_point_cloud)

    render_option = vis.get_render_option()
    render_option.line_width = 5.0
    vis.run()
    vis.destroy_window()

    #o3d.visualization.draw_geometries([pcd, graph_lines, pcd_house])

def visualize_graph_import(vis, point_cloud_file: str, node_cloud_file: str, connections_file: str):
    pcd = load_point_cloud_from_pcl(node_cloud_file)
    print("Loaded point cloud")

    # Load edges
    edges = load_edges_from_file(connections_file)
    print("Loaded edges")

    # Create graph lines
    graph_lines = create_graph_edges(edges, offst=True)
    print("Created edges")

    pcd_house = load_point_cloud_from_pcl(point_cloud_file)

    vis.add_geometry(pcd)
    vis.add_geometry(graph_lines)

    visualize_graph_import.point_cloud_visible = False
    def toggle_point_cloud(vis):
        if not visualize_graph_import.point_cloud_visible:
            vis.add_geometry(pcd_house, reset_bounding_box=False)
            print("adding geom")
        else:
            vis.remove_geometry(pcd_house, reset_bounding_box=False)
            print("removing geom")
        visualize_graph_import.point_cloud_visible = not visualize_graph_import.point_cloud_visible
        return False  # Stops the event propagation

    vis.register_key_callback(32, toggle_point_cloud)
    render_option = vis.get_render_option()
    render_option.line_width = 5.0
    return vis



if __name__ == "__main__":
    pcd_filename = "dense_18.pcd"
    nodes_cloud_file = "nodes_obj.pcd"  # replace with your .pcd file
    connections_file = "connections_obj.txt"
    visualize_graph(pcd_filename, nodes_cloud_file, connections_file)
    
