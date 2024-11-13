import random
import numpy as np
import open3d as o3d

def load_nodes(file_path):
    """Loads nodes from a .pcd file using Open3D and returns a list of coordinates."""
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points).tolist()

def load_adjacency_matrix(file_path):
    """Load adjacency matrix from file and return as numpy array."""
    return np.loadtxt(file_path, dtype=int)

def generate_waypoints(adjacency_matrix, nodes, num_waypoints):
    """Generate a sequence of waypoints based on adjacency and nodes."""
    node_count = len(nodes)
    start_node = random.randint(0, node_count - 1)
    path = [start_node]
    
    for _ in range(num_waypoints - 1):
        neighbors = np.where(adjacency_matrix[path[-1]] == 1)[0]
        if not neighbors.size:
            break  # Stop if there are no further connections.
        next_node = random.choice(neighbors)
        path.append(next_node)
        
    return [nodes[node] for node in path]

def generate_command_file(num_humans, waypoints_per_human, nodes_file, adj_matrix_file, output_file):
    nodes = load_nodes(nodes_file)
    adjacency_matrix = load_adjacency_matrix(adj_matrix_file)

    with open(output_file, 'w') as f:
        commands = ""
        for i in range(num_humans):
            # Generate a random character name.
            character_name = f"Character{i+1}"
            
            # Randomly generate a path for each human.
            waypoints = generate_waypoints(adjacency_matrix, nodes, waypoints_per_human)
            
            # Command to spawn the character at the first waypoint.
            x, y, z = waypoints[0]
            f.write(f"Spawn {character_name} {x} {y} 0 0\n")

            """
            queue_name = f"{character_name}_Queue"
            queue_command = f"Queue {queue_name}\n"
            for i, (x, y, z) in enumerate(waypoints[1:]):
                queue_command += f"Queue_Spot {queue_name} {i} {x} {y} 0 _\n"
            f.write(queue_command)

            reverse_queue_name = f"{character_name}_ReverseQueue"
            reverse_queue_command = f"Queue {reverse_queue_name}\n"
            for i, (x, y, z) in enumerate(reversed(waypoints[:-1])):
                reverse_queue_command += f"Queue_Spot {reverse_queue_name} {i} {x} {y} 0 _\n"
            f.write(reverse_queue_command)
            """
            goto_command = f"{character_name} GoTo"
            for x, y, z in waypoints[1:]:
                goto_command += f" {x} {y} 0"
            goto_command += f" _\n"
            commands += goto_command

            """
            lookaround_command = f"{character_name} LookAround {random.randint(0, 10)}\n"
            commands += lookaround_command

            goback_command = f"{character_name} GoTo"
            for x, y, z in reversed(waypoints[:-1]):
                goback_command += f" {x} {y} 0"
            goback_command += f" _\n"
            commands += goback_command
            """
        f.write(commands)

# Parameters
num_humans = 10  # Example: number of humans
waypoints_per_human = 20  # Example: waypoints per human
nodes_file = '/workspace/data_proc/data19/graph_experiments/nodes/nodes_Ekhoe_3_05.pcd'  # Path to nodes file
adj_matrix_file = '/workspace/data_proc/data19/graph_experiments/connections/adjacency_matrix_Ekhoe_3_05.txt'  # Path to adjacency matrix file
output_file = '/workspace/data_proc/data19/human_simulation_commands.txt'  # Output command file path

# Run the generator
generate_command_file(num_humans, waypoints_per_human, nodes_file, adj_matrix_file, output_file)
