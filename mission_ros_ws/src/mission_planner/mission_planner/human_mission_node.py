import argparse
import rclpy
from rclpy.node import Node

import json
import time

import py_trees
from mission_planner_interfaces.srv import HumanQuery, QueryGoal, FindPath
from std_msgs.msg import String
from rclpy.executors import MultiThreadedExecutor

class MissionPlanner(Node):
    def __init__(self, query="sofa"):
        super().__init__('human_query')

        # create query service 
        self.query_srv = self.create_service(
            HumanQuery, 
            '/human_query', 
            self.service_callback
        )

        self.result_ready = False
        self.response = None

        self.cf_query_future = None
        self.path_query_future = None

        self.timer_node = rclpy.create_node('timer_node')
        self.process_timer = self.timer_node.create_timer(
            0.5, 
            self.process
        )
        self.query_client = self.timer_node.create_client(
            QueryGoal, 
             "/cf_tools/query_goal"
        )
        self.find_path_client = self.timer_node.create_client(
            FindPath, 
            "/global_planner/find_path"
        )

        self.query_result_pub = self.create_publisher(
            String, 
            "/query_result", 
            10
        )


    def service_callback(self, req, res):
        self.get_logger().info(f"Received query: {req.query}")
        req_json = json.loads(req.query)
        response = {
            "query": req_json.get("query", ""),
            "source": req_json.get("source", {}),
            "target": [],
            "path": [],
            "node_id": -1,
            "node_description": "",
        }

        assert "x" in response["source"] and "y" in response["source"], "Location must have x and y coordinates"
        if "z" not in response["source"]:
            location["z"] = 1.0
        assert response["query"], "Query string must not be empty"
        
        self.result_ready = False
        self.response = response
        while not self.result_ready:
            time.sleep(0.1)
        response = self.response
        self.response = None
        
        vis_result = String()
        vis_result.data = json.dumps(response)
        self.query_result_pub.publish(vis_result)
        self.get_logger().info(f"Published query result: {vis_result.data}")

        res.response = json.dumps(response)
        return res


    def mock_process(self):
        if self.response is None or self.result_ready:
            return

        if self.response["target"] == []:
            self.response["target"] = {
                "x": 3.0,
                "y": 1.0,
                "z": 1.0
            }
            self.response["node_id"] = 1
            self.response["node_description"] = "sofa"

        else:
            self.response["path"] = [
                {"x": 0.0, "y": 0.0, "z": 1.0},
                {"x": 1.0, "y": 0.0, "z": 1.0},
                {"x": 1.0, "y": 1.0, "z": 1.0},
                {"x": 3.0, "y": 1.0, "z": 1.0}
            ]
            self.result_ready = True
    

    def process(self):
        if self.response is None or self.result_ready:
            return

        # work with cf
        if self.response["target"] == []:

            if self.cf_query_future is None:
                self.cf_query_future = self.query_goal(self.response["query"])

            print(self.cf_query_future.result())
            
            if self.cf_query_future.done():
                self.get_logger().info(f"Found: {self.cf_query_future.result()}")

                # Query failed beacuse of ROS
                if self.cf_query_future.result() is None:
                    self.result_ready = True
                    return

                cf_response = json.loads(self.cf_query_future.result().response)
                node_id = cf_response["object_id"]

                # Query failed because of LLM
                if node_id == -1:
                    self.get_logger().info("No relevant nodes found")
                    self.result_ready = True
                    return
                
                node_description = cf_response["object_desc"]
                self.get_logger().info(f"Query successful, found: {node_description}")
                node_location = cf_response["location"]

                self.response["target"] = node_location
                self.response["node_id"] = node_id
                self.response["node_description"] = node_description

        # work with global planner
        else:
            if self.path_query_future is None:
                self.path_query_future = self.find_path(self.response["source"], self.response["target"])

            if self.path_query_future.done():
                if self.path_query_future.result() is None:
                    self.get_logger().error("Find path service failed")
                    self.result_ready = True
                    return

                self.get_logger().info(f"Found path to object")
                path = json.loads(self.path_query_future.result().response)
                self.response["path"] = path
                self.result_ready = True


    # Query language model for target position
    def query_goal(self, query):
        if not self.query_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f'Query service not available')
            res.response = json.dumps(response)
            return res
        self.get_logger().info(f"Querying object: {query}")
        request_json = json.dumps({"query_string": query})
        request = QueryGoal.Request(query=request_json)
        future = self.query_client.call_async(request)
        return future


    # Query global planner for waypoints
    def find_path(self, location, node_location):
        if not self.find_path_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f'Path finding service not available.')
            res.response = json.dumps(response)
            return res

        query = json.dumps({
            "start": location,
            "target": node_location
        })
        self.get_logger().info(f"Finding path to object: {query}")
        request = FindPath.Request(query=query)
        future = self.find_path_client.call_async(request)
        return future


def main(args=None):
    rclpy.init(args=args)
    node = MissionPlanner()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.add_node(node.timer_node)
    executor.spin()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
