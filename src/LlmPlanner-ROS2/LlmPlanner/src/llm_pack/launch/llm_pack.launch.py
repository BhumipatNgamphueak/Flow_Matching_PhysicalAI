#!/usr/bin/python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, LaunchContext
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription,  OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import launch.logging

def generate_launch_description():

    launch_description = LaunchDescription()
    package_name = 'llm_pack'

    llm_node = Node(
        package=package_name,
        executable="llm_node.py"
    )

    # Add the actions to the launch description 
    launch_description.add_action(llm_node)

    return launch_description