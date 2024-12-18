from cf_compact.utils.slam_classes import MapObjectList


def load_map_objects(map_objects_file):
    map_objects = MapObjectList()
    map_objects.load(map_objects_file)
    return map_objects


def gen_samples_on_unit_sphere(n_samples=100):
    # Fibonacci sphere
    sampling_directions = []
    phi = np.pi * (3 - np.sqrt(5))  # Golden angle in radians

    for i in range(n_samples):
        y = 1 - 2 * (i / (n_samples - 1))  # Map i to range [-1, 1]
        radius = np.sqrt(1 - y * y)  # Compute the radius of the circle at height y
        theta = phi * i  # Angle for the point
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        sample = np.array([x, y, z])
        sample_directions.append(sample)
        
    return sample_directions

def gen_samples_on_unit_circle(n_samples=100):
    # Regular circle
    sampling_directions = []
    phi = 2 * np.pi  # Full circle in radians

    for i in range(n_samples):
        theta = phi * (i / n_samples)  # Angle for each sample
        x = np.cos(theta)
        y = np.sin(theta)

        sample = np.array([x, y, 0.0])
        sample_directions.append(sample)
    
    return sample_directions