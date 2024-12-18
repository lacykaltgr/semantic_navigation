class FindNextBestView(py_trees.behaviour.Behaviour):
    def __init__(self, node):
        super(WaypointHandler, self).__init__(name="WaypointHandler")
        self.node = node

        # Attach blackboard
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key="path", access=READ)
        self.blackboard.register_key(key="next_waypoint", access=WRITE)
        
        self.future = None
        
    def setup(self):
        self.node.get_logger().info("Setting up WaypointHandler...")
        # setup ros clients

    def initialise(self):
        self.node.get_logger().info("Initializing WaypointHandler...")
        # called when previous next best view is reached
        self.future = None

    def update(self):
        if self.future is None:
            # trigger query, save future
            # ask mapping server for current next best view
            return RUNNING
        
        if self.future.done():
            successful = True
            if successful:
                return SUCCESS
            else:
                return FAILURE
        else:
            return RUNNING
            
