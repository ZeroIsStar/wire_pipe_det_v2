cmake_minimum_required(VERSION 3.8)
project(wireandpipe_detection_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)

set(ONNXRUNTIME_INCLUDE_DIR /usr/local/include)
set(ONNXRUNTIME_LIB_DIR /usr/local/lib)

add_executable(wireandpipe_detection_node
  src/wireandpipe_detection.cpp
)

target_include_directories(wireandpipe_detection_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${ONNXRUNTIME_INCLUDE_DIR}
)

ament_target_dependencies(wireandpipe_detection_node
  builtin_interfaces
  geometry_msgs
  nav_msgs
  rclcpp
  sensor_msgs
  std_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
  visualization_msgs
)

target_link_libraries(wireandpipe_detection_node
  ${OpenCV_LIBRARIES}
  ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so
)

install(TARGETS wireandpipe_detection_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(FILES config/params.yaml
  DESTINATION share/${PROJECT_NAME}/config
)

install(FILES src/7_6.onnx
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
