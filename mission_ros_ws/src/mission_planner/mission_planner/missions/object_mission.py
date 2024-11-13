import rclpy
import py_trees
import json
import tf2_ros
from tf2_ros.transform_listener import TransformListener
from std_msgs.msg import String

from .waypoint_mission import WaypointMission
from mission_planner_interfaces.srv import QueryGoal, FindPath
from . import READ, WRITE, SUCCESS, FAILURE, RUNNING


class WhatIsTheQuery(py_trees.behaviour.Behaviour):
    def __init__(self, node):
        super(WhatIsTheQuery, self).__init__(name="WhatIsTheQuery")
        self.node = node

        # blackboard
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key="query_lifo", access=READ)
        self.blackboard.register_key(key="query", access=WRITE)


    def update(self):
        n_queries = len(self.blackboard.query_lifo)
        if n_queries == 0:
            self.node.get_logger().info("No queries in LIFO")
            return RUNNING
        query = self.blackboard.query_lifo.pop()
        self.blackboard.query = query
        self.node.get_logger().info(f"Query: {query}")
        return SUCCESS


# behaviour for querying objects
class QueryObject(py_trees.behaviour.Behaviour):
    def __init__(self, node):
        super(QueryObject, self).__init__(name="QueryObject")
        self.node = node

        # blackboard
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key="object_id", access=WRITE)
        self.blackboard.register_key(key="object_desc", access=WRITE)
        self.blackboard.register_key(key="object_location", access=WRITE)
        self.blackboard.register_key(key="query", access=READ)

        self.client_name = "/cf_tools/query_goal"
        self.query_goal_client = self.node.create_client(QueryGoal, self.client_name)
        self.query_result_pub = self.node.create_publisher(String, "/query_result", 10)


    def update(self):
        if not self.query_goal_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().error(f'Service not available: {self.client_name}')
            return FAILURE

        self.node.get_logger().info(f"Querying object: {self.blackboard.query}")
        request_json = json.dumps({"query_string": self.blackboard.query})
        request = QueryGoal.Request(query=request_json)
        future = self.query_goal_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)
        
        if future.result() is not None:
            response = json.loads(future.result().response)
            self.node.get_logger().info(f"Query response: {response['object_desc']}")
            self.query_result_pub.publish(String(data=json.dumps(response)))
            object_id = response["object_id"]
            if object_id == -1:
                self.node.get_logger().info("No object found")
                return FAILURE
            # Write the query result to the blackboard
            self.blackboard.object_id = object_id
            self.blackboard.object_desc = response["object_desc"]
            self.blackboard.object_location = response["location"]
 
            return SUCCESS
        else:
            self.node.get_logger().error("Query service failed")
            return FAILURE
        

# behaviour to find location of object
class FindRobot(py_trees.behaviour.Behaviour):
    def __init__(self, node):
        super(FindRobot, self).__init__(name="FindRobot")
        self.node = node

        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key="robot_location", access=WRITE)

        self.tf_buffer = tf2_ros.Buffer(node=self.node)
        self.tf_listener = TransformListener(self.tf_buffer, self.node)
        

    def update(self):
        try:
            trans = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))
            trans = trans.transform.translation
            robot_location = {"x": trans.x, "y": trans.y, "z": trans.z}
            self.blackboard.robot_location = robot_location
            self.node.get_logger().info(
                f"Robot at x: {robot_location}"
            )
            return SUCCESS
        except Exception as e:
            self.node.get_logger().error(f"Transform lookup failed: {e}")
            return FAILURE

'''
Not using this for now
# behaviour to find location of object
class FindObject(py_trees.behaviour.Behaviour):
    def __init__(self, node):
        super(FindObject, self).__init__(name="MoveToObject")
        self.node = node

    def setup(self):
        self.tf_buffer = tf2_ros.Buffer(node=self.node)

    def initialise(self):
        # Read object information from the blackboard
        blackboard = Blackboard()
        self.object_id = blackboard.get("object_id", None)
        self.object_desc = blackboard.get("object_desc", None)

        if self.object_id is None or self.object_desc is None:
            self.node.get_logger().error("Object ID or Description missing from Blackboard")
        else:
            self.node.get_logger().info(f"Finding object: {self.object_desc}")

    def update(self):
        # Check if the object_id and object_desc are valid
        if self.object_id is None or self.object_desc is None:
            return py_trees.common.Status.FAILURE

        try:
            trans = self.tf_buffer.lookup_transform("map", "object_location", rclpy.time.Time())
            Blackboard().object_location = trans
            self.node.get_logger().info(f"Object {self.object_desc} at x: {trans.transform.translation.x}, y: {trans.transform.translation.y}")
            return py_trees.common.Status.SUCCESS
        except Exception as e:
            self.node.get_logger().error(f"Transform lookup failed: {e}")
            return py_trees.common.Status.FAILURE
'''

# behaviour to find global path to object
class FindPathToObject(py_trees.behaviour.Behaviour):
    def __init__(self, node):
        super(FindPathToObject, self).__init__(name="FindPathToObject")
        self.node = node

        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key="path", access=WRITE)
        self.blackboard.register_key(key="object_location", access=READ)
        self.blackboard.register_key(key="robot_location", access=READ)

        self.client_name = "/global_planner/find_path"
        self.find_path_client = self.node.create_client(FindPath, self.client_name)

        """
        self.mock_response = {
            "path": [
                {"x": -0.2, "y": -3.0, "z": 0.0},
                {"x": -2.0, "y": -3.0, "z": 0.0},
                {"x": -3.0, "y": 0.0, "z": 0.0},
            ]
        }
        """


    def prep_query(self):
        self.node.get_logger().info("Finding path to object...")
        robot_location = self.blackboard.robot_location
        object_location = self.blackboard.object_location
        return json.dumps({
            "start": robot_location,
            "target": object_location
        })

    def update(self):
        if not self.find_path_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().error(f'Service not available: {self.client_name}')
            return FAILURE

        query = self.prep_query()
        self.node.get_logger().info(f"finding path to object: {query}")
        request = FindPath.Request(query=query)

        # Mock response
        #self.blackboard.path = self.mock_response
        #return SUCCESS
        

        future = self.find_path_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)
        
        if future.result() is not None:
            self.node.get_logger().info(f"Found path to object")
            path = json.loads(future.result().response)
            self.blackboard.path = path
            return SUCCESS
        else:
            self.node.get_logger().error("Query service failed")
            return FAILURE



class ObjectMission(py_trees.composites.Sequence):
    def __init__(self, node, query):
        self.node = node
        self.query = query
        super().__init__(
            name="ObjectMission",
            memory=True,
            children= [
                QueryObject(self.node, self.query),
                FindRobot(self.node),
                #FindObject(self.node),
                FindPathToObject(self.node),
                WaypointMission(self.node)
            ]
        )

