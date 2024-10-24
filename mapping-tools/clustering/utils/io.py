import json
import open3d as o3d
import numpy as np
import networkx as nx


# TXT
# ----

def read_txt_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    return content

def write_txt_lines(lines, output_file):
    with open(output_file, 'w') as f:
        for line in lines:
            f.write(f"{line}\n")

def read_txt_lines(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        lines[i] = line.strip()
    return lines


# JSON
# ----


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


# PCD
# ----

def load_pcd(pcd_file):
    pcd = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(pcd.points)
    print(f"Number of points in the point cloud: {len(points)}")
    return points

def create_point_cloud(points):
    points = np.array(points)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud

def save_point_cloud(point_cloud, output_file):
    o3d.io.write_point_cloud(output_file, point_cloud)


# Graph


def load_adjacency_matrix(adj_file):
    adjacency_matrix = np.loadtxt(adj_file)
    print(f"Adjacency matrix shape: {adjacency_matrix.shape}")
    G = nx.from_numpy_array(adjacency_matrix)
    return G



