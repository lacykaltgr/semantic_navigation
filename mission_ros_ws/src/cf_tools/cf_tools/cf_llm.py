import os
import numpy as np
import torch
import argparse
from PIL import Image
import json
from groq import Groq

import rclpy
from rclpy.node import Node
from mission_planner_interfaces.srv import QueryGoal
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import CompressedImage
from .mapobjectlist import MapObjectList


from .utils import read_json_file, query_groq

class ConceptGraphTools(Node):
    def __init__(self):
        super().__init__('cf_llm')

        self.init_gpt()

        # Services
        self.query_goal_srv = self.create_service(QueryGoal, 'cf_tools/query_goal', self.query_goal)

        self.get_logger().info("cf_llm: SERVICES READY!")


    def query_goal(self, req, res):
        req_json = json.loads(req.query)
        query = req_json.get("query_string", "")
        excluded_ids = [] #req.get("excluded_ids", [])

        id_, desc, location = self.ask_gpt(query, excluded_ids)

        location = {
            "x": location[0],
            "y": location[1],
            "z": location[2]
        }

        res.response = json.dumps({
            "object_id": id_,
            "object_desc": desc,
            "location": location,
        })
        return res


    def init_gpt(self):
        self.declare_parameters(
            namespace='',
            parameters=[
                ('system_prompt_path', '/app/mission_ros_ws/src/cf_tools/resource/prompts/cf_planner_prompt_cluster.txt'),
                ('scene_json_path', '/workspace/data_proc/data19/scene_description.json'),
            ]
        )
        system_prompt_path = self.get_parameter("system_prompt_path").value
        scene_json_path = self.get_parameter("scene_json_path").value

        # Load System Prompt and JSON scene description
        self.client = Groq()
        self.system_prompt = open(system_prompt_path, "r").read()
        scene = read_json_file(scene_json_path)

        self.global_context = scene["global_context"]
        clusters = scene["clusters"]
        n_clusters = len(clusters)
        print(f"Loaded {n_clusters} clusters from {scene_json_path}")

        n_objects = 0
        scene_desc_prompt = []
        locations = {}

        for cluster_id, cluster in enumerate(clusters):
            n_objects += len(cluster["objects"])
            objects_prompt = []
            for obj in cluster["objects"]:
                if obj["object_tag"] in ["invalid", "none", "unknown"]:
                    continue
                object_id = f"obj_{n_objects}"
                n_objects += 1
                objects_prompt.append({
                    "object_id": object_id,
                    "object_tag": obj["object_tag"],
                })

                locations.update(    
                    {object_id: obj["bbox_center"]}
                )

            scene_desc_prompt.append({
                "cluster_id": f"cluster_{cluster_id}",
                "label": cluster["label"],
                "description": cluster["description"],
                "objects": objects_prompt,
            })
            locations.update(
                {f"cluster_{cluster_id}": cluster["location"]}
            )

        self.scene_desc = scene_desc_prompt
        self.locations = locations

        print(f"Loaded {n_objects} objects from {scene_json_path}")
        

    def ask_gpt(self, query, excluded_ids=[]):
        
        #if len(excluded_ids):
        #    scene = [o for o in self.scene_desc if o["id"] not in excluded_ids]
        #else:
        #    scene = self.scene_desc

        response = query_groq(query, self.system_prompt, self.scene_desc, self.global_context, self.client)
        self.get_logger().info("GPT Response")

        query_achievable = response["query_achievable"]

        if query_achievable:
            id_ = response["final_relevant"][0]
            desc = response["most_relevant_desc"]

            # Find object data
            location = self.locations[id_]
        else:
            id_, desc = -1, "NOT AN OBJECT"
            location = [0, 0, 0]

        return id_, desc, location



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
    rclpy.spin(node)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
