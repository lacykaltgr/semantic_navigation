import argparse
import rclpy
from rclpy.node import Node

import py_trees
from mission_planner_interfaces.srv import HumanQuery, QueryGoal

class MissionPlanner(Node):
    def __init__(self, query="sofa"):
        super().__init__('human_query')

        # create query service 
        self.query_srv = self.create_service(
            HumanQuery, 
            '/human_query', 
            self.query_goal,
            10
        )
        self.query_client = self.create_client(
            QueryGoal, 
             "/cf_tools/query_goal"
        )
        self.find_path_client = self.node.create_client(
            FindPath, 
            "/global_planner/find_path"
        )
        

    def query_goal(self, req, res):
        req_json = json.loads(req.query)
        location = req_json.get("source", {})
        query = req_json.get("query_string", "")

        assert "x" in location and "y" in location, "Location must have x and y coordinates"
        if "z" not in location:
            location["z"] = 1.0
        assert query, "Query string must not be empty"

        response = {
            "source": location,
            "target": [],
            "path": [],
            "node_id": -1,
            "node_desciption": "",
        }

        if not self.query_client.wait_for_service(timeout_sec=5.0):
            self.node.get_logger().error(f'Query service not available: {self.client_name}')
            res.response = json.dumps(response)
            return res


        # Query language model for target position

        self.node.get_logger().info(f"Querying object: {query}")
        request_json = json.dumps({"query_string": query})
        request = QueryGoal.Request(query=request_json)
        future = self.query_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)

        if future.result() is None:
            res.response = json.dumps(response)
            return res

        response = json.loads(future.result().response)
        node_id = response["object_id"]

        if node_id == -1:
            self.node.get_logger().info("No relevant nodes found")
            res.response = json.dumps(response)
            return res
        
        node_description = response["object_desc"]
        self.node.get_logger().info(f"Query successful, found: {node_description}")
        node_location = response["location"]

        res["target"] = node_location
        res["node_id"] = node_id
        res["node_description"] = node_description


        # Query global planner for waypoints

        if not self.find_path_client.wait_for_service(timeout_sec=5.0):
            self.node.get_logger().error(f'Path finding service not available: {self.client_name}')
            res.response = json.dumps(response)
            return res

        query = json.dumps({
            "start": location,
            "target": node_location
        })
        self.node.get_logger().info(f"Finding path to object: {query}")
        request = FindPath.Request(query=query)

        future = self.find_path_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)
        
        if future.result() is None:
            self.node.get_logger().error("Find path service failed")
            res.response = json.dumps(response)
            return res

        self.node.get_logger().info(f"Found path to object")
        path = json.loads(future.result().response)
        res["path"] = path
        res.response = json.dumps(response)

        return res


def main(args=None):
    rclpy.init(args=args)
    node = MissionPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
