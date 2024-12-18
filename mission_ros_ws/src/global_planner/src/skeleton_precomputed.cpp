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

    void loadMap(
        const std::string &config_path = "",
        const std::string &nodes_path = "",
        const std::string &edges_path = ""
    ) {

        YAML::Node config = YAML::LoadFile(config_path);
        skeleton_finder = std::make_unique<SkeletonFinder>(config);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PCDReader reader;
        reader.read(nodes_path, *cloud);

        skeleton_finder->loadNodes(cloud);
        skeleton_finder->loadConnections(edges_path);
        cout << skeleton_finder->getNodes().size() << " nodes loaded" << endl;
        cout << "Skeleton finder initialized..." << endl;
    }

public:
    rclcpp::Service<FindPath>::SharedPtr find_path_service_;

    GlobalPlannerNode(const std::string &config_path = "") : Node("global_planner_node") {
        this->declare_parameter<std::string>("config_path", config_path);
        this->declare_parameter<std::string>("nodes_path", "");
        this->declare_parameter<std::string>("edges_path", "");
        
        string _config_path = this->get_parameter("config_path").as_string();
        string _nodes_path = this->get_parameter("nodes_path").as_string();
        string _edges_path = this->get_parameter("edges_path").as_string();

        loadMap(_config_path, _nodes_path, _edges_path);
        
        // add service to find path
        find_path_service_ = this->create_service<FindPath>(
            "/global_planner/find_path", std::bind(&GlobalPlannerNode::handleFindPath, this, std::placeholders::_1, std::placeholders::_2)
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

        auto path_w_radius = findPath(start_x, start_y, start_z, target_x, target_y, target_z);
        vector<Eigen::Vector3d> path = path_w_radius.first;
        vector<double> radiuses = path_w_radius.second;
        
        // no radius is returned for the target node
        //assert (path.size() == radiuses.size() + 1);
        // add radius for the target node (should be handled by cf_tools)
        radiuses.push_back(0.5);
        radiuses.push_back(0.2);
        

        //convert path to json
        nlohmann::json path_json;
        for (size_t i = 0; i < path.size(); i++) {
            Eigen::Vector3d point = path[i];
            double radius = radiuses[i];
            nlohmann::json point_json;
            point_json["x"] = point(0);
            point_json["y"] = point(1);
            point_json["z"] = point(2);
            point_json["radius"] = radius;
            path_json.push_back(point_json);
        }

        response->response = path_json.dump();
    }


    pair<vector<Eigen::Vector3d>, vector<double>> findPath(
        double start_x, double start_y, double start_z, 
        double target_x, double target_y, double target_z
    ) {
        return skeleton_finder->run_findpath(start_x, start_y, start_z, target_x, target_y, target_z);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);

    try {
        auto node = std::make_shared<GlobalPlannerNode>();
        rclcpp::spin(node);
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    rclcpp::shutdown();
    return 0;
}
