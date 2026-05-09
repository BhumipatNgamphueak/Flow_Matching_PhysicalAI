"""
turtlesim_trapezoid.launch.py
------------------------------
Launches turtlesim_plus + turtle_controller + turtle_plotter + llm_node
(LLM planner that publishes constraints on /traj_context).

Select the controller mode with  mode:=trapezoid  or  mode:=pid.
Disable the LLM node with  use_llm:=false  if you only want manual /goal_position.

── Trapezoid mode (classical trapezoidal velocity profile) ──────────────
  ros2 launch trajectory_publisher turtlesim_trapezoid.launch.py mode:=trapezoid

── PID/CFM mode (flow matching trajectory + PID tracker) ────────────────
  ros2 launch trajectory_publisher turtlesim_trapezoid.launch.py mode:=pid

── Issue an LLM prompt at runtime ───────────────────────────────────────
  ros2 service call /LlmPrompt llm_pack_interface/srv/String \\
      "{prompt: 'go to x=2.0, y=1.5 with v=0.15 and a=0.03'}"

── Required env var (LLM only) ──────────────────────────────────────────
  export GOOGLE_API_KEY=...
  ros2 launch trajectory_publisher turtlesim_trapezoid.launch.py mode:=pid
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ── declare arguments ─────────────────────────────────────────────────
    args = [
        # common
        DeclareLaunchArgument('mode',          default_value='trapezoid',
                              description="Control mode: 'trapezoid' or 'pid'"),
        DeclareLaunchArgument('turtle_name',   default_value='turtle1',
                              description='Name of the turtle to control'),
        DeclareLaunchArgument('start_delay',   default_value='2.0',
                              description='Seconds to wait before moving'),
        DeclareLaunchArgument('use_llm',       default_value='true',
                              description='Launch the LLM planner node (requires GOOGLE_API_KEY)'),

        # trapezoid (fallback if no /traj_context)
        DeclareLaunchArgument('v_max',         default_value='2.0',
                              description='[trapezoid] Fallback max linear velocity (m/s)'),
        DeclareLaunchArgument('a_max',         default_value='0.5',
                              description='[trapezoid] Fallback max acceleration (m/s²)'),
        DeclareLaunchArgument('distance',      default_value='5.0',
                              description='[trapezoid] Fallback travel distance (m)'),

        # PID (fallback goal if no /traj_context, plus PID gains)
        DeclareLaunchArgument('goal_x',        default_value='10.0',
                              description='[pid] Fallback target X (m, world)'),
        DeclareLaunchArgument('goal_y',        default_value='7.5',
                              description='[pid] Fallback target Y (m, world)'),
        DeclareLaunchArgument('goal_tolerance',default_value='0.05',
                              description='[pid] Arrival tolerance (m)'),
        DeclareLaunchArgument('kp_linear',     default_value='1.5',
                              description='[pid] Linear PID  Kp'),
        DeclareLaunchArgument('ki_linear',     default_value='0.0',
                              description='[pid] Linear PID  Ki'),
        DeclareLaunchArgument('kd_linear',     default_value='0.05',
                              description='[pid] Linear PID  Kd'),
        DeclareLaunchArgument('kp_angular',    default_value='5.0',
                              description='[pid] Angular PID Kp'),
        DeclareLaunchArgument('ki_angular',    default_value='0.0',
                              description='[pid] Angular PID Ki'),
        DeclareLaunchArgument('kd_angular',    default_value='0.1',
                              description='[pid] Angular PID Kd'),
        DeclareLaunchArgument('v_max_pid',     default_value='0.20',
                              description='[pid] Linear velocity clamp (m/s) — match CFM training range [0.10, 0.20]'),
        DeclareLaunchArgument('w_max_pid',     default_value='3.0',
                              description='[pid] Angular velocity clamp (rad/s)'),
    ]

    # ── turtlesim_plus simulator ──────────────────────────────────────────
    turtlesim_node = Node(
        package='turtlesim_plus',
        executable='turtlesim_plus_node.py',
        name='turtlesim_plus',
        output='screen',
    )

    # ── unified controller ────────────────────────────────────────────────
    controller_node = Node(
        package='trajectory_publisher',
        executable='turtle_controller.py',
        name='turtle_controller',
        output='screen',
        parameters=[{
            'mode':           LaunchConfiguration('mode'),
            'turtle_name':    LaunchConfiguration('turtle_name'),
            'start_delay':    LaunchConfiguration('start_delay'),
            # trapezoid fallbacks
            'v_max':          LaunchConfiguration('v_max'),
            'a_max':          LaunchConfiguration('a_max'),
            'distance':       LaunchConfiguration('distance'),
            # pid
            'goal_x':         LaunchConfiguration('goal_x'),
            'goal_y':         LaunchConfiguration('goal_y'),
            'goal_tolerance': LaunchConfiguration('goal_tolerance'),
            'kp_linear':      LaunchConfiguration('kp_linear'),
            'ki_linear':      LaunchConfiguration('ki_linear'),
            'kd_linear':      LaunchConfiguration('kd_linear'),
            'kp_angular':     LaunchConfiguration('kp_angular'),
            'ki_angular':     LaunchConfiguration('ki_angular'),
            'kd_angular':     LaunchConfiguration('kd_angular'),
            'v_max_pid':      LaunchConfiguration('v_max_pid'),
            'w_max_pid':      LaunchConfiguration('w_max_pid'),
        }],
    )

    # ── real-time plotter ─────────────────────────────────────────────────
    plotter_node = Node(
        package='trajectory_publisher',
        executable='turtle_plotter.py',
        name='turtle_plotter',
        output='screen',
        parameters=[{
            'turtle_name': LaunchConfiguration('turtle_name'),
        }],
    )

    # ── LLM planner node (Gemini via GOOGLE_API_KEY) ──────────────────────
    # Inherits GOOGLE_API_KEY from the launching shell.
    llm_node = Node(
        package='llm_pack',
        executable='llm_node.py',
        name='llm_node',
        output='screen',
        additional_env={
            'GOOGLE_API_KEY': os.environ.get('GOOGLE_API_KEY', ''),
        },
        condition=IfCondition(LaunchConfiguration('use_llm')),
    )

    # Delay controller + plotter so simulator's turtle1 topics are ready
    delayed_local_nodes = TimerAction(
        period=1.5,
        actions=[controller_node, plotter_node],
    )

    # Delay LLM node a bit longer so the /traj_context subscriber is up first
    delayed_llm = TimerAction(
        period=3.0,
        actions=[llm_node],
    )

    return LaunchDescription(args + [turtlesim_node, delayed_local_nodes, delayed_llm])
