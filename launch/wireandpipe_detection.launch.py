#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# export WIRE_PIPE_AVOID_HOLD_SECONDS=5.0
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, EnvironmentVariable
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory("wireandpipe_detection_cpp")

    default_params_file = os.path.join(pkg_share, "config", "params.yaml")
    default_model_path = os.path.join(pkg_share, "only_pipe_det.onnx")

    params_file = LaunchConfiguration("params_file", default=default_params_file)
    avoid_hold_seconds = LaunchConfiguration("avoid_hold_seconds")

    declare_params_file = DeclareLaunchArgument(
        "params_file",
        default_value=default_params_file,
        description="Full path to the ROS parameters file for wire/pipe detection",
    )

    declare_avoid_hold_seconds = DeclareLaunchArgument(
        "avoid_hold_seconds",
        default_value=EnvironmentVariable("WIRE_PIPE_AVOID_HOLD_SECONDS", default_value="3.0"),
        description="Seconds to hold avoidance state after obstacle disappears",
    )

    wireandpipe_detection_node = Node(
        package="wireandpipe_detection_cpp",
        executable="wireandpipe_detection_node",
        name="wireandpipe_detection_node",
        output="screen",
        parameters=[
            params_file,
            {"yolo_model_path": default_model_path},
            {"avoid_hold_seconds": LaunchConfiguration("avoid_hold_seconds")},
        ],
        respawn=True,
        respawn_delay=10.0,
    )

    return LaunchDescription([
        declare_params_file,
        declare_avoid_hold_seconds,
        wireandpipe_detection_node,
    ])