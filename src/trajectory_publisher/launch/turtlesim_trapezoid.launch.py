"""
turtlesim_trapezoid.launch.py
------------------------------
Launches turtlesim_plus + turtle_controller + turtle_plotter.

Select the controller mode with  mode:=trapezoid  or  mode:=pid.

── Trapezoid mode examples ────────────────────────────────────────────────────
  ros2 launch trajectory_publisher turtlesim_trapezoid.launch.py
  ros2 launch trajectory_publisher turtlesim_trapezoid.launch.py v_max:=3.0 distance:=8.0

── PID mode examples ──────────────────────────────────────────────────────────
  ros2 launch trajectory_publisher turtlesim_trapezoid.launch.py mode:=pid
  ros2 launch trajectory_publisher turtlesim_trapezoid.launch.py mode:=pid goal_x:=12.0 goal_y:=10.0

── All launch arguments ───────────────────────────────────────────────────────
  Common
    mode          'trapezoid' | 'pid'       default: trapezoid
    turtle_name   name of the turtle        default: turtle1
    start_delay   seconds before moving     default: 2.0

  Trapezoid-only
    v_max         max linear velocity m/s   default: 2.0
    a_max         max acceleration  m/s²    default: 0.5
    distance      travel distance m         default: 5.0

  PID-only
    goal_x        target X position m       default: 10.0
    goal_y        target Y position m       default: 7.5
    goal_tolerance  stop radius m           default: 0.05
    kp_linear     PID linear  Kp            default: 1.5
    ki_linear     PID linear  Ki            default: 0.0
    kd_linear     PID linear  Kd            default: 0.05
    kp_angular    PID angular Kp            default: 5.0
    ki_angular    PID angular Ki            default: 0.0
    kd_angular    PID angular Kd            default: 0.1
    v_max_pid     linear  velocity clamp    default: 2.0
    w_max_pid     angular velocity clamp    default: 3.0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
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

        # trapezoid
        DeclareLaunchArgument('v_max',         default_value='2.0',
                              description='[trapezoid] Max linear velocity (m/s)'),
        DeclareLaunchArgument('a_max',         default_value='0.5',
                              description='[trapezoid] Max acceleration (m/s²)'),
        DeclareLaunchArgument('distance',      default_value='5.0',
                              description='[trapezoid] Travel distance (m)'),

        # PID
        DeclareLaunchArgument('goal_x',        default_value='10.0',
                              description='[pid] Target X position (m)'),
        DeclareLaunchArgument('goal_y',        default_value='7.5',
                              description='[pid] Target Y position (m)'),
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
        DeclareLaunchArgument('v_max_pid',     default_value='2.0',
                              description='[pid] Linear velocity clamp (m/s)'),
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
            # trapezoid
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

    # Delay controller + plotter so simulator's turtle1 topics are ready
    delayed_nodes = TimerAction(
        period=1.5,
        actions=[controller_node, plotter_node],
    )

    return LaunchDescription(args + [turtlesim_node, delayed_nodes])
