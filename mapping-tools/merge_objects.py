# Merge multiple results from conceptfusion-compact
import sys
import pickle
import gzip
import os
import open3d as o3d
import numpy as np

from cf_compact.utils.slam_classes import MapObjectList
from cf_compact.utils.general_utils import measure_time, save_pointcloud, save_obj_json, ObjectClasses
from cf_compact.utils.slam_utils import denoise_objects, filter_objects, merge_objects

def load_result(result_path):
    potential_path = os.path.realpath(result_path)

    if potential_path != result_path:
        print(f"Resolved symlink for result_path: {result_path} -> \n{potential_path}")
        result_path = potential_path
    with gzip.open(result_path, "rb") as f:
        print(f"Loading results from {result_path}")
        results = pickle.load(f)
    if not isinstance(results, dict):
        raise ValueError("Results should be a dictionary! other types are not supported!")

    objects = MapObjectList()
    objects.load_serializable(results["objects"])
    # not using bg_objects for now
    bg_objects = []
    #bg_objects.extend(obj for obj in objects if obj['is_background'])
    class_colors = results['class_colors']
    config = results['cfg']
    print("Objects loaded successfully.")

    if len(bg_objects) == 0:
        bg_objects = None
    return objects, bg_objects, class_colors, config


def process_objects(objects, cfg):
    objects = measure_time(denoise_objects)(
        downsample_voxel_size=cfg['downsample_voxel_size'], 
        dbscan_remove_noise=cfg['dbscan_remove_noise'], 
        dbscan_eps=cfg['dbscan_eps'], 
        dbscan_min_points=cfg['dbscan_min_points'], 
        spatial_sim_type=cfg['spatial_sim_type'], 
        device=cfg['device'], 
        objects=objects
    )

    objects = filter_objects(
        obj_min_points=cfg['obj_min_points'], 
        obj_min_detections=cfg['obj_min_detections'], 
        objects=objects,
    )

    objects = measure_time(merge_objects)(
        merge_overlap_thresh=0.3,#cfg["merge_overlap_thresh"],
        merge_visual_sim_thresh=0.3,#cfg["merge_visual_sim_thresh"],
        merge_text_sim_thresh=0.3, #cfg["merge_text_sim_thresh"],
        objects=objects,
        downsample_voxel_size=0.1, #cfg["downsample_voxel_size"],
        dbscan_remove_noise=cfg["dbscan_remove_noise"],
        dbscan_eps=cfg["dbscan_eps"],
        dbscan_min_points=cfg["dbscan_min_points"],
        spatial_sim_type=cfg["spatial_sim_type"],
        device=cfg["device"],
        do_edges=False,
    )
    return objects

if __name__ == "__main__":
    # Example usage : python merge_objects.py [result_path1] [result_path2] ... [out_dir_path]

    all_objects = MapObjectList()
    class_color = None
    config = None

    result_paths = sys.argv[1:-1]
    out_dir_path = sys.argv[-1]

    for path in result_paths:
        objects, _, class_color, cfg = load_result(path)
        all_objects.extend(objects)
        all_objects = process_objects(all_objects, cfg)

    obj_classes = ObjectClasses(
        classes_file_path="/app/conceptfusion-compact/cf_compact/classes/scannet200_classes.txt", 
        bg_classes=cfg['bg_classes'], 
        skip_bg=cfg['skip_bg']
    )

    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)

    save_pointcloud(
        exp_suffix="merged",
        exp_out_path=out_dir_path,
        cfg=cfg,
        objects=all_objects,
        obj_classes=obj_classes,
        latest_pcd_filepath=None,
        create_symlink=False,
    )

    save_obj_json(
        exp_suffix="merged",
        exp_out_path=out_dir_path,
        objects=all_objects
    )

    merged_pcd = o3d.geometry.PointCloud()
    for obj in all_objects:
        obj_pcd = obj["pcd"]

        obj_classes = np.array(obj["class_id"])
        values, counts = np.unique(obj_classes, return_counts=True)
        obj_class_id = values[np.argmax(counts)]
        # set color
        obj_pcd.colors = o3d.utility.Vector3dVector(
            np.array([class_color[str(obj_class_id)] for _ in range(len(obj_pcd.points))])
        )
        merged_pcd += obj["pcd"]
    # save merged pcd
    o3d.io.write_point_cloud(os.path.join(out_dir_path, "obj_merged.pcd"), merged_pcd)



"""
1 pkllgz
2 json - bounding boxes
3 pcd with class colors - points, colors

"""

    