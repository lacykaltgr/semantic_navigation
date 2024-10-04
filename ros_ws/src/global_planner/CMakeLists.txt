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


add_executable(global_planner 
  src/global_planner.cpp
)

ament_target_dependencies(global_planner
  rclcpp
  mission_planner_interfaces
)

target_link_libraries(global_planner
  SkeletonFinder
  Eigen3::Eigen
  ${PCL_LIBRARIES}  
  yaml-cpp
)

install(TARGETS
  global_planner
  DESTINATION lib/${PROJECT_NAME})

ament_package()
