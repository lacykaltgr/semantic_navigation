cmake_minimum_required(VERSION 3.8)
project(global_planner)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(SkeletonFinder REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(mission_planner_interfaces REQUIRED)

# Add executables
add_executable(skeleton_mapped 
  src/skeleton_mapped.cpp
)

add_executable(skeleton_precomputed
  src/skeleton_precomputed.cpp
)

# Link dependencies for both nodes
ament_target_dependencies(skeleton_mapped
  rclcpp
  mission_planner_interfaces
)

ament_target_dependencies(skeleton_precomputed
  rclcpp
  mission_planner_interfaces
)

target_link_libraries(skeleton_mapped
  SkeletonFinder
  Eigen3::Eigen
  ${PCL_LIBRARIES}  
  yaml-cpp
)

target_link_libraries(skeleton_precomputed
  SkeletonFinder
  Eigen3::Eigen
  ${PCL_LIBRARIES}  
  yaml-cpp
)

# Install targets
install(TARGETS
  skeleton_mapped
  skeleton_precomputed
  DESTINATION lib/${PROJECT_NAME})

ament_package()