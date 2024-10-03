#include <rclcpp/rclcpp.hpp>
#include <iostream>
#include <string>

#include <SkeletonFinder/skeleton_finder_3D.h>
#include <math.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sys/time.h>
#include <time.h>
#include <pcl/search/impl/kdtree.hpp>
#include <fstream>
#include <sstream>
#include <vector>
#include <mission_planner_interfaces/srv/find_path.hpp>
#include <nlohmann/json.hpp>

using namespace std;
typedef mission_planner_interfaces::srv::FindPath FindPath;


class GlobalPlannerNode : public rclcpp::Node {

private:
    std::unique_ptr<SkeletonFinder> skeleton_finder;

    void loadMap(const std::string &map_path = "", const std::string &config_path = "") {

        YAML::Node config = YAML::LoadFile(config_path);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

        pcl::PCDReader reader;
        reader.read(map_path, *cloud);

        cout << "Map successfully loaded..." << endl;
        cout << "Size of map: " << (*cloud).points.size() << endl;

        skeleton_finder = std::make_unique<SkeletonFinder>(config);

        cout << "Skeleton finder initialized..." << endl;

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::copyPointCloud(*cloud, *cloud_xyz);

        skeleton_finder->run_processing(cloud_xyz);
    }

public:
    rclcpp::Service<FindPath>::SharedPtr find_path_service_;

    GlobalPlannerNode(const std::string &map_dir_path = "", const std::string &config_path = "") : Node("global_planner_node") {
        this->declare_parameter("map_path", map_dir_path);
        this->declare_parameter("config_path", config_path);

        string _map_path = this->get_parameter("map_path").as_string();
        string _config_path = this->get_parameter("config_path").as_string();

        loadMap(_map_path, _config_path);
        
        // add service to find path
        find_path_service_ = this->create_service<FindPath>(
            "/global_planner/find_path", std::bind(&GlobalPlannerNode::handleFindPath, this, std::placeholders::_1, std::placeholders::_2)
        );

        nearest_node_service = this->create_service<NearestNode>(
            "/global_planner/nearest_node", std::bind(&GlobalPlannerNode::handleNearestNode, this, std::placeholders::_1, std::placeholders::_2)
        );

    }


    void handleFindPath(const std::shared_ptr<FindPath::Request> request,
                        std::shared_ptr<FindPath::Response> response) {
        //parse request string for json
        nlohmann::json rq_json = nlohmann::json::parse(request->query);
        //parse json for start and target
        double start_x = rq_json["start"]["x"];
        double start_y = rq_json["start"]["y"];
        double start_z = rq_json["start"]["z"];
        double target_x = rq_json["target"]["x"];
        double target_y = rq_json["target"]["y"];
        double target_z = rq_json["target"]["z"];

        vector<Eigen::Vector3d> path = findPath(start_x, start_y, start_z, target_x, target_y, target_z);

        //convert path to json
        nlohmann::json path_json;
        for (Eigen::Vector3d point : path) {
            nlohmann::json point_json;
            point_json["x"] = point(0);
            point_json["y"] = point(1);
            point_json["z"] = point(2);
            path_json.push_back(point_json);
        }

        response->response = path_json.dump();
    }

    void handleNearestNode(const std::shared_ptr<FindPath::Request> request,
                        std::shared_ptr<FindPath::Response> response) {

        vector<NodeNearestNeighbors> ngbs = skeleton_finder->run_nearestnodes();

        //convert path to json
        nlohmann::json ngbs_json;
        for (NodeNearestNeighbors ngb : ngbs) {
            nlohmann::json node_json;
            point_json["idx"] = ngb.index;
            point_json["coord"] = [ngb.coord1, ngb.coord2, ngb.coord3];
            point_json["neighbors"] = ngb.nearest_nodes;    
            path_json.push_back(point_json);
        }

        response->response = path_json.dump();
    }

    vector<Eigen::Vector3d> findPath(
        double start_x, double start_y, double start_z, 
        double target_x, double target_y, double target_z
    ) {
        return skeleton_finder->run_findpath(start_x, start_y, start_z, target_x, target_y, target_z);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);

    std::string map_path;
    std::string config_path;

    // Argument parsing
    if (argc < 2) {
        //std::cerr << "Error: The path cannot be empty." << std::endl;
        //return 1;
        map_path = "/workspace/ros2_ws/src/global_planner/libraries/polygon_generation/resource/dense_18.pcd";
        config_path = "/workspace/ros2_ws/src/global_planner/libraries/polygon_generation/config.yaml";
    } else {
        map_path = argv[1];
        config_path = argv[2];
    }


    auto node = std::make_shared<GlobalPlannerNode>(map_path, config_path);
    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}
