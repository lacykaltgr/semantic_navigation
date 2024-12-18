# input: 
origins = [np.array([0, 0, 0]), np.array([1, 1, 1]), np.array([2, 2, 2])]
features_dict = {
    "feature1": Feature1(weight=0.5),
    "feature2": Feature2(weight=0.3),
    "feature3": Feature3(weight=0.2),
}
num_rays = 1000
num_views = 10


evaluator = RayCastingEvaluator(features_dict, view_dim=ViewDim.VIEW3D, num_views=num_views)

for origin in origins:
    view_ratings = evaluator.evaluate(origin, num_rays)
    print(ratings)
