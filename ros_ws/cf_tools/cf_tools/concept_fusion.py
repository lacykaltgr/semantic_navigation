import os
import numpy as np
import torch
import argparse
from PIL import Image
import json

import rclpy
from rclpy.node import Node
from mission_planner_interfaces.srv import QueryGoal
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import CompressedImage
from .mapobjectlist import MapObjectList


from .utils import (
    read_json_file, 
    query_llm, 
    query_clip
)

class ConceptGraphTools(Node):
    def __init__(self):
        super().__init__('cf_tools')

        self.declare_parameter(
            'query_model', 'clip'
        )
        self.query_model = self.get_parameter('query_model').value

        if self.query_model == 'clip':
            self.init_clip()
            self.ask_model = self.ask_clip
        elif self.query_model == 'gpt':
            self.init_gpt()
            self.ask_model = self.ask_gpt
        else:
            raise ValueError(f"Invalid query model: {self.query_model}")

        # Services
        self.query_goal_srv = self.create_service(QueryGoal, 'cf_tools/query_goal', self.query_goal)

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

    def spin(self):
        rclpy.spin(self)

    def on_shutdown(self):
        pass

    def init_gpt(self):
        self.declare_parameters(
            namespace='',
            parameters=[
                ('system_prompt_path', './prompts/concept_graphs_planner.txt'),
                ('scene_json_path', './scene.json'),
            ]
        )
        system_prompt_path = self.get_parameter("system_prompt_path").value
        scene_json_path = self.get_parameter("scene_json_path").value

        # Load System Prompt and JSON scene description
        self.system_prompt = open(system_prompt_path, "r").read()
        self.scene_desc = read_json_file(scene_json_path)

        # Filter out some objects
        n_objects = len(self.scene_desc)
        self.scene_desc = [o for o in self.scene_desc if o["object_tag"] not in ["invalid", "none", "unknown"]]
        new_n_objects = len(self.scene_desc)
        self.get_logger().info(
            f"Removed {n_objects - new_n_objects} objects from the scene with invalid tags. {new_n_objects} left."
        )

        # Remove possible_tags
        for o in self.scene_desc:
            o.pop("bbox_extent")
        


    def ask_gpt(self, query, excluded_ids=[]):
        
        if len(excluded_ids):
            scene = [o for o in self.scene_desc if o["id"] not in excluded_ids]
        else:
            scene = self.scene_desc

        response = query_llm(query, self.system_prompt, scene)
        self.get_logger().info("GPT Response")
        self.get_logger().info(response)

        query_achievable = response["query_achievable"]

        if query_achievable:
            object_id = response["final_relevant_objects"][0]
            object_desc = response["most_relevant_desc"]

            # Find object data
            for object_data in self.scene_desc:
                if object_data["id"] == object_id:
                    break
            location = object_data["bbox_center"]
        else:
            object_id, object_desc = -1, "NOT AN OBJECT"
            location = [0, 0, 0]

        return object_id, object_desc, location
    

    def init_clip(self):
        import open_clip
        import pickle
        import gzip

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
        obj_id, obj_desc, location = query_clip(query, self.objects, self.clip_tokenizer, self.clip_model)

        #location_message = Float32MultiArray(data=location)
        #self.location_publisher.publish(location_message)

        return obj_id, obj_desc, location



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
