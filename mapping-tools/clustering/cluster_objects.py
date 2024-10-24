import os
import argparse
import numpy as np
import networkx as nx
from scipy.spatial import KDTree

try:
    from utils.llm_queries import query_groq_describe_image, query_groq_label_cluster
    from utils.io import read_txt_file, read_txt_lines, load_json, load_pcd, load_adjacency_matrix, create_point_cloud, save_point_cloud, save_json
except ImportError:
    from .utils.llm_queries import query_groq_describe_image, query_groq_label_cluster
    from .utils.io import read_txt_file, read_txt_lines, load_json, load_pcd, load_adjacency_matrix, create_point_cloud, save_point_cloud, save_json


def get_graph_largest_component(G, points=None):
    # Get the largest connected component
    largest_component = max(nx.connected_components(G), key=len)

    if len(largest_component) == G.number_of_nodes():
        print("Graph is already fully connected.")
        return G, points
    
    print("Graph is not connected. Filtering to the largest connected component.")
    G_largest = G.subgraph(largest_component).copy()
    print(f"Number of nodes in the largest connected component: {G_largest.number_of_nodes()}")
    if points is None:
        return G_largest

    # Filter the point cloud to keep only points in the largest connected component
    filtered_points = points[list(largest_component)]
    print(f"Filtered number of points: {len(filtered_points)}")
    return G_largest, filtered_points


def cluster_graph(G, clustering_algorithm='louvain', num_clusters=8):

    # Clustering
    if clustering_algorithm == 'louvain':
        communities = nx.community.louvain_communities(G, seed=123)
    elif clustering_algorithm == 'girvan_newman':
        communities = list(nx.community.girvan_newman(G))[-1]
    elif clustering_algorithm == 'asyn_fluidc':
        communities = list(nx.community.asyn_fluidc(G, num_clusters))
    elif clustering_algorithm == 'asyn_lpa':
        communities = list(nx.community.asyn_lpa_communities(G))
    elif clustering_algorithm == 'greedy_modularity':
        communities = list(nx.community.greedy_modularity_communities(G))
    else:
        raise ValueError("Unsupported clustering algorithm")
    
    print(f"Number of clusters: {len(communities)}")

    communities_dict = {idx: list(community) for idx, community in enumerate(communities)}
    return communities_dict


def get_image_descriptions(G, communities, points, poses, pose_image_paths, prompt, n_image_queries=3):
    poses_kdtree = KDTree(poses)
    descriptions = dict()
    
    for cluster_idx, cluster in communities.items():
        cluster_subgraph = G.subgraph(cluster).copy()
        # calculate edge betweenness centrality of each node
        edge_betweenness = nx.betweenness_centrality(cluster_subgraph)
        # find the n_nodes_per_cluster nodes with the highest edge betweenness centrality
        dominant_nodes = sorted(edge_betweenness, key=edge_betweenness.get, reverse=True)[:n_image_queries]

        closest_poses = []
        for node in dominant_nodes:
            distance, index = poses_kdtree.query(points[node])
            closest_poses.append(int(index))

        image_paths = [pose_image_paths[pose] for pose in closest_poses]

        # ask language model
        descriptions[cluster_idx] = []
        for image_path in image_paths:
            descriptions_json = query_groq_describe_image(prompt, image_path)
            descriptions[cluster_idx].append(descriptions_json)

    return descriptions



def label_clusters(G, communities, points, descriptions, system_prompt, global_context):
    # Create a KDTree for efficient nearest neighbor search
    kdtree = KDTree(points)
    
    # Assign objects to clusters based on proximity to cluster points
    clusters_with_objects = []
    for cluster_idx, community in communities.items():
        cluster_info = {
            "cluster_id": cluster_idx,
            "objects": [],
            "image_descriptions": descriptions[cluster_idx]
        }
        
        # TODO: should do this elsewhere?
        # Check each object and find the closest point in the current cluster
        for obj in objects.values():
            bbox_center = np.array(obj['bbox_center'])
            distances, index = kdtree.query(bbox_center)
            nearest_point_index = list(G.nodes())[int(index)]
            if nearest_point_index in community:
                cluster_info["objects"].append(obj)
        
        clusters_with_objects.append(cluster_info)

    
    # Ask the language model for a description of each cluster
    response = query_groq_label_cluster(
        clusters_with_objects, system_prompt, global_context
    )

    for cluster in response["clusters"]:
        cluster_id = cluster["cluster_id"]
        label = cluster["label"]
        description = cluster["description"]

        clusters_with_objects[int(cluster_id)]["label"] = label
        clusters_with_objects[int(cluster_id)]["description"] = description

    scene_desc = {
        "global_context": global_context,
        "clusters": clusters_with_objects
    }
    
    return scene_desc


if __name__ == '__main__':
    # Parser
    parser = argparse.ArgumentParser(description='Cluster skeleton graph and label each cluster.')
    parser.add_argument('data_dir', type=str, help='Path to the data folder')
    parser.add_argument('--clustering_algorithm', type=str, default='louvain', help='Clustering algorithm to use')
    parser.add_argument('--num_clusters', type=int, default=8, help='Number of clusters to generate')
    parser.add_argument('--n_image_queries', type=int, default=3, help='Number of image queries per cluster')
    parser.add_argument('--global_context_file', type=str, default='./assets/global_context.txt', help='Path to the global context file')
    parser.add_argument('--label_clusters_prompt_file', type=str, default='./assets/label_cluster_prompt.txt', help='Path to the system prompt file')
    parser.add_argument('--describe_image_prompt_file', type=str, default='./assets/describe_image_prompt.txt', help='Path to the system prompt file')
    parser.add_argument('--communities_output_file', type=str, default='communities.json', help='Path to the communities output JSON file')
    parser.add_argument('--description_output_file', type=str, default='cluster_descriptions.json', help='Path to the descriptions output JSON file')
    parser.add_argument('--output_file', type=str, default='scene_description.json', help='Path to the output JSON file')
    args = parser.parse_args()

    data_dir = args.data_dir
    pcd_file = os.path.join(data_dir, 'nodes_demo.pcd')
    adj_file = os.path.join(data_dir, 'adjacency_matrix.txt')
    poses_pcd_file = os.path.join(data_dir, 'poses.pcd')
    pose_image_paths_file = os.path.join(data_dir, 'poses_paths.txt')
    obj_json_file = os.path.join(data_dir, 'obj_json_merged.json')
    global_context_file = os.path.realpath(args.global_context_file)
    label_clusters_prompt_file = os.path.realpath(args.label_clusters_prompt_file)
    describe_image_prompt_file = os.path.realpath(args.describe_image_prompt_file)

    communities_output_file = os.path.join(data_dir, args.communities_output_file)
    description_output_file = os.path.join(data_dir, args.description_output_file)
    output_file = os.path.join(data_dir, args.output_file)

    # Load
    G = load_adjacency_matrix(adj_file)
    points = load_pcd(pcd_file)
    # Extract the largest connected component if not connected
    G_largest, filtered_points = get_graph_largest_component(G, points)


    # Clustering
    if not os.path.exists(communities_output_file):
        print("Clustering the graph...")
        communities = cluster_graph(G_largest, args.clustering_algorithm, args.num_clusters)
        print("Saving clustering to file: ", communities_output_file)
        save_json(communities, communities_output_file)
    else:
        print("Clustering already exists. Loading from file.")
        communities = load_json(communities_output_file)

    # Descriptions
    global_context = read_txt_file(global_context_file)
    if not os.path.exists(description_output_file):
        print("Querying language model for image descriptions...")
        poses = load_pcd(poses_pcd_file)
        pose_image_paths = read_txt_lines(pose_image_paths_file)
        describe_image_prompt = read_txt_file(describe_image_prompt_file) 
        describe_image_prompt = describe_image_prompt + global_context
        descriptions = get_image_descriptions(G_largest, communities, points, poses, pose_image_paths, describe_image_prompt, args.n_image_queries)
        print("Saving descriptions to file: ", description_output_file)
        save_json(descriptions, description_output_file)
    else:
        print("Descriptions already exist. Loading from file.")
        descriptions = load_json(description_output_file)

    # Cluster labeling
    objects = load_json(obj_json_file)
    label_clusters_prompt = read_txt_file(label_clusters_prompt_file)
    scene_desc = label_clusters(G_largest, communities, filtered_points, descriptions, label_clusters_prompt, global_context)
    print("Saving scene description to file: ", output_file)
    save_json(scene_desc, output_file)
