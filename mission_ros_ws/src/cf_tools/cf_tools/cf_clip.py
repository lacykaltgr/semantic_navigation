import os
import numpy as np
import torch
import argparse
from PIL import Image
import json
import open_clip
import pickle
import gzip

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import CompressedImage
from mission_planner_interfaces.srv import QueryGoal

from .mapobjectlist import MapObjectList
from .utils import read_json_file, clip_similarities

class ConceptGraphTools(Node):
    def __init__(self):
        super().__init__('cf_tools')

        self.init_clip()
        self.ask_model = self.ask_clip

        # Services
        self.similarity_service = self.create_service(QueryGoal, 'cf_clip/similarity', self.query_goal)
        self.query_goal_srv = self.create_service(QueryGoal, 'cf_tools/query_goal', self.similarity_service)

        self.get_logger().info("cf_tools: SERVICES READY!")


    def query_goal(self, req, res):
        req_json = json.loads(req.query)
        query = req_json.get("query_string", "")
        excluded_ids = [] #req.get("excluded_ids", [])

        object_id, object_desc, location = self.ask_model(query, excluded_ids)

        location = {
            "x": location[0],
            "y": location[1],
            "z": location[2]
        }

        res.response = json.dumps({
            "object_id": object_id,
            "object_desc": object_desc,
            "location": location,
        })
        return res

    def similarity_service():
        req_json = json.loads(req.query)
        query = req_json.get("query_string", "")
        excluded_ids = [] #req.get("excluded_ids", [])

        clip_probs = self.clip_probs(query)

        results = dict()
        for obj, prob in zip(self.objects, clip_probs):
            results[obj["id"]] = float(prob)

        res.response = json.dumps(results)
        return res
        

    def init_clip(self):
        self.declare_parameter(
            'result_path', '/workspace/ros2_ws/src/cf_tools/resource/pcd_mapping.pkl.gz'
        )
        result_path = self.get_parameter("result_path").value

        print("Loading objects...")
        with gzip.open(result_path, "rb") as f:
            print(f"Loading results from {result_path}")
            results = pickle.load(f)
        objects = MapObjectList()
        objects.load_serializable(results["objects"])
        self.objects = objects
        print("Done loading objects.")

        print("Initializing CLIP model...")
        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
        self.clip_model = clip_model #.to("cuda")
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
        print("Done initializing CLIP model.")


    def ask_clip(self, query, excluded_ids=[]):
        similarities = clip_similarities(query, self.objects, self.clip_tokenizer, self.clip_model)

        probs = F.softmax(similarities, dim=0)
        max_prob_idx = torch.argmax(probs).item()
        max_prob_object = objects[max_prob_idx]
        print(f"Most probable object is at index {max_prob_idx} with class name '{max_prob_object['class_name']}'")
        print(f"location xyz: {max_prob_object['bbox'].center}")

        return max_prob_idx, max_prob_object['class_name'], max_prob_object['bbox'].center


    def clip_probs(self, query):
        similarities = clip_similarities(query, self.objects, self.clip_tokenizer, self.clip_model)

        # calculate noramlized similarity scores
        max_value = similarities.max()
        min_value = similarities.min()
        normalized_similarities = (similarities - min_value) / (max_value - min_value)

        probs = F.softmax(similarities, dim=0)
        return probs.detach().numpy()

    def spin(self):
        rclpy.spin(self)

    def on_shutdown(self):
        pass



def main(args=None):
    rclpy.init(args=args)

    # Argparse with a system_prompt_path and a scene_json_path argument
    #parser = argparse.ArgumentParser()
    #parser.add_argument("-p", "--system_prompt_path", type=str, default="prompts/concept_graphs_planner.txt")
    #parser.add_argument("-s", "--scene_json_path", type=str, default="scene.json")
    #parser.add_argument("-t", "--sim_threshold", type=float, default=0.26)
    #parser.add_argument("-o", "--object_detector", type=str, default="llava_full_image")
    #args = parser.parse_args()

    node = ConceptGraphTools()
    node.spin()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
