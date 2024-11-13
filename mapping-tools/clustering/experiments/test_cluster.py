import os
import json  # Make sure to import json for saving/loading
import numpy as np
import networkx as nx
from scipy.spatial import KDTree
from sklearn.cluster import SpectralClustering
from infomap import Infomap  # Import the Infomap package
from tqdm import tqdm


####################################
# Utils
####################################

def load_adjacency_matrix(adj_file):
    adjacency_matrix = np.loadtxt(adj_file)
    print(f"Adjacency matrix shape: {adjacency_matrix.shape}")
    G = nx.from_numpy_array(adjacency_matrix)
    return G

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


####################################
# Clustering
####################################

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

def cluster_graph(G, clustering_algorithm='louvain', num_clusters=10):
    print(f"Clustering algorithm: {clustering_algorithm}")
    
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
    elif clustering_algorithm == 'edge_betweenness':
        communities = list(nx.community.edge_betweenness_partition(G, num_clusters))
    elif clustering_algorithm == 'infomap':
        # Initialize Infomap with default settings
        infomap_instance = Infomap("--two-level")
        
        # Add edges to Infomap instance
        for u, v in G.edges():
            infomap_instance.add_link(u, v)
        
        # Run the Infomap algorithm
        infomap_instance.run()
        
        # Retrieve communities
        communities_dict = {}
        for node in infomap_instance.tree:
            if node.is_leaf:
                communities_dict.setdefault(node.module_id, []).append(node.node_id)
                
        communities = list(communities_dict.values())
    elif clustering_algorithm == 'spectral':
        adjacency_matrix = nx.to_numpy_array(G)
        spectral = SpectralClustering(n_clusters=num_clusters, affinity='precomputed')
        labels = spectral.fit_predict(adjacency_matrix)
        communities_dict = {i: [] for i in range(num_clusters)}
        for node, label in enumerate(labels):
            communities_dict[label].append(node)
        communities = communities_dict.values()
    else:
        raise ValueError("Unsupported clustering algorithm")

    print(f"Number of clusters: {len(communities)}")
    print(communities)
    communities_dict = {idx: list(community) for idx, community in enumerate(communities)}
    
    return communities_dict


if __name__ == '__main__':

    data_dir = "/workspace/data_proc/data19/graph_experiments"

    experiments = [
        #"grid_0.500000",
        "random_2000_2",
        "Ekhoe_3_05",
        "Skeleton"
    ]

    cluster_algo_list = [
        "louvain",
        "edge_betweenness",
        "infomap",
        "spectral",
        "asyn_fluidc",
    ]

    output_dir = os.path.join(data_dir, 'clustering')
    os.makedirs(output_dir, exist_ok=True)
    adjacency_matrices = [f"{data_dir}/connections/adjacency_matrix_{exp}.txt" for exp in experiments]

    for adjexp, exp in tqdm(zip(adjacency_matrices, experiments)):
        G = load_adjacency_matrix(adjexp)
        G_largest = get_graph_largest_component(G)[0]
        print(G_largest)
        for cluster_algo in cluster_algo_list:
            communities = cluster_graph(G_largest, clustering_algorithm=cluster_algo)
            output_file = f"{output_dir}/communities_{exp}_{cluster_algo}.json"
            save_json(communities, output_file)



