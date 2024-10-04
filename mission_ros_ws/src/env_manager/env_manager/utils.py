import gymnasium as gym
import numpy as np

def space_to_dict(space):
    if isinstance(space, gym.spaces.Box):
        return {
            "type": "Box",
            "low": space.low.tolist(),
            "high": space.high.tolist(),
            "shape": space.shape,
            "dtype": str(space.dtype)
        }
    elif isinstance(space, gym.spaces.Discrete):
        return {
            "type": "Discrete",
            "n": space.n
        }
    # Handle other space types as needed
    else:
        return str(space)  # Fallback for unknown types


def dict_np2list(d):
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            d[key] = value.tolist()
        elif isinstance(value, dict):
            d[key] = dict_np2list(value)
    return d