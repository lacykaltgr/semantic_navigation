cmake_minimum_required(VERSION 3.5)
project(mission_planner_interfaces)

#set(PYTHON_EXECUTABLE /opt/conda/envs/mission_planner_env/bin/python)
#message(STATUS "Using Python interpreter: ${PYTHON_EXECUTABLE}")


# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(action_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rosidl_typesupport_c REQUIRED)
find_package(rosidl_typesupport_cpp REQUIRED)
find_package(rosidl_generator_py REQUIRED)  # Python bindings
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)


rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/String.msg"
  "msg/MoveTo.msg"
  "msg/EnvStepAction.msg"
  "srv/Strings.srv"
  "srv/QueryGoal.srv"
  "srv/FindPath.srv"
  "srv/HumanQuery.srv"
  "action/MoveTo.action"
  DEPENDENCIES std_msgs action_msgs std_srvs
 )


ament_export_dependencies(rosidl_default_runtime)


ament_package()