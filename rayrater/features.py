from rayrater_classes import RayFeature

class LLMFeature(RayFeature):
    def __init__(self, weight, objects, prompt_file):
        super().__init__("LLM Rater", (1), weight)
        
    def compute_feature_value(self, ray):
        return np.dot(self.feature_vector, ray.direction)
    
    def __repr__(self):
        return f"LLMFeature(ray={self.ray}, feature_vector={self.feature_vector})"
    
    
class ConfidenceFeature(RayFeature):
    def __init__(self, weight, objects, prompt_file):
        super().__init__("Confidence Rater", (1), weight)
        
    def compute_feature_value(self, ray):
        return np.dot(self.feature_vector, ray.direction)
    
    def __repr__(self):
        return f"ConfidenceFeature(ray={self.ray}, feature_vector={self.feature_vector})"