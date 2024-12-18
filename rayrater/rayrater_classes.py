from abc import ABC, abstractmethod

from utils import gen_samples_on_unit_sphere, gen_samples_on_unit_circle

class RayFeature(ABC):
    def __init__(self, name, weight):
        self.name = name
        self.weight = weight

    @abstractmethod
    # return a list containing the feature values for each ray
    def compute_values(self, rays):
        pass
    
    def compute_weighted_values(self, rays):
        return self.weight * self.compute_values(rays)
    
    
class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction
    
    def get_2d_projection(self, up_axis=2):
        projected_ray = self.direction.copy()
        projected_ray[up_axis] = 0
        norm = np.linalg.norm(projected_ray)
        if norm != 0:
            projected_ray /= norm
        return projected_ray


class ViewDim(Enum):
    VIEW2D = 1
    VIEW3D = 2 


class View:
    
    def __init__(self, direction, features_dict, view_dim, fov):
        self.direction = direction
        self.features_dict = features_dict
        self.view_dim = view_dim
        self.fov = fov
        self.fov_cosine_threshold = np.cos(np.radians(fov / 2))
          
    def get_rays_mask(self, rays):
        rays = np.zeros(len(rays))
        for i, ray in enumerate(rays):
            if self.sees_ray(ray):
                rays[i] = 1
        return rays
        
    def assign_ray(self, ray: Ray):
        if self.view_dim == ViewDim.VIEW2D:
            ray_direction = ray.get_2d_projection()
        else:
            ray_direction = ray.direction
        cos_theta = np.dot(ray_direction, self.direction)
        sees_ray = cos_theta > self.fov_cosine_threshold:
        return sees_ray



class ViewEvaluator:
    
    def __init__(self, features_dict, view_type=None, num_views=10, fov=None):
        self.features_dict = features_dict
        if (view_dim is None):
            view_dim = ViewDim.VIEW3D
        self.view_dim = view_dim
        self.num_views = num_views
        
        if view_type == ViewDim.VIEW3D:
            self.view_directions = gen_samples_on_unit_sphere(num_views)
        elif view_type == ViewDim.VIEW2D:
            self.view_directions = gen_samples_on_unit_sphere(num_views) 
        else:
            raise ValueError("Invalid view type")
        self.views = [View(direction, features_dict, view_dim, fov) 
                      for direction in self.view_directions]
    
    def evaluate(self, origin, num_rays, return_dict=False):
        ray_directions = gen_samples_on_unit_sphere(num_rays)
        rays = [Ray(origin, direction) for direction in ray_directions]
        ray_ratings = self.get_ray_ratings(rays)
        view_ratings = []
        for view in self.views:
            view_mask = view.get_rays_mask(rays)
            view_value = np.sum(view_mask * ray_ratings)
            view_ratings.append(view_value)
        return view_ratings

        
    def evaluate_feature(self, feature_name, return_dict=False):
        values_dict = {}
        for view in self.views:
            values_dict[view] = view.get_feature_value(feature_name)
        return values_dict
    
    def get_ray_ratings(self, rays: List[Ray], return_dict=False):
        ratings = np.zeros(len(rays))
        ratings_dict = {} if return_dict else None
        for feat_name, feat in self.features_dict.items():
            feature_rating = feature.compute_weighted_values(rays)
            if return_dict:
                ratings_dict[feat_name] = feature_rating
            ratings += feature_rating
        if return_dict:
            return ratings, ratings_dict
        return ratings
    
    def get_ray_rating_for_feature(self, rays, feature_name, weighted=False):
        feature = self.features_dict[feature_name]
        if weighted:
            return feature.compute_weighted_values(rays)
        return feature.compute_values(rays)


