"""Launch the NanoDet camera detector with resource-conscious defaults."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory("nanodet_ros2")
    config_file = os.path.join(package_share, "config", "detector.yaml")

    arguments = [
        DeclareLaunchArgument("model_path", default_value=""),
        DeclareLaunchArgument("model_metadata_path", default_value=""),
        DeclareLaunchArgument("input_topic", default_value="/camera/image_raw"),
        DeclareLaunchArgument(
            "detections_topic", default_value="/nanodet/detections"
        ),
        DeclareLaunchArgument("annotated_topic", default_value="/nanodet/image"),
        DeclareLaunchArgument("confidence_threshold", default_value="0.4"),
        DeclareLaunchArgument("nms_threshold", default_value="0.6"),
        DeclareLaunchArgument("max_rate_hz", default_value="5.0"),
        DeclareLaunchArgument("input_reliability", default_value="best_effort"),
        DeclareLaunchArgument("output_reliability", default_value="best_effort"),
        DeclareLaunchArgument("allowed_labels", default_value=""),
        DeclareLaunchArgument("runtime", default_value="auto"),
        DeclareLaunchArgument("runtime_threads", default_value="4"),
        DeclareLaunchArgument("runtime_allow_spinning", default_value="false"),
        DeclareLaunchArgument("publish_annotated", default_value="true"),
    ]

    detector = Node(
        package="nanodet_ros2",
        executable="detector_node",
        name="nanodet_detector",
        output="screen",
        parameters=[
            config_file,
            {
                "model_path": LaunchConfiguration("model_path"),
                "model_metadata_path": LaunchConfiguration("model_metadata_path"),
                "input_topic": LaunchConfiguration("input_topic"),
                "detections_topic": LaunchConfiguration("detections_topic"),
                "annotated_topic": LaunchConfiguration("annotated_topic"),
                "confidence_threshold": ParameterValue(
                    LaunchConfiguration("confidence_threshold"), value_type=float
                ),
                "nms_threshold": ParameterValue(
                    LaunchConfiguration("nms_threshold"), value_type=float
                ),
                "max_rate_hz": ParameterValue(
                    LaunchConfiguration("max_rate_hz"), value_type=float
                ),
                "input_reliability": LaunchConfiguration("input_reliability"),
                "output_reliability": LaunchConfiguration("output_reliability"),
                "allowed_labels": LaunchConfiguration("allowed_labels"),
                "runtime": LaunchConfiguration("runtime"),
                "runtime_threads": ParameterValue(
                    LaunchConfiguration("runtime_threads"), value_type=int
                ),
                "runtime_allow_spinning": ParameterValue(
                    LaunchConfiguration("runtime_allow_spinning"), value_type=bool
                ),
                "publish_annotated": ParameterValue(
                    LaunchConfiguration("publish_annotated"), value_type=bool
                ),
            },
        ],
    )

    return LaunchDescription(arguments + [detector])
