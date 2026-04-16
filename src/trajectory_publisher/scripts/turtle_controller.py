#!/usr/bin/python3

"""
turtle_controller.py
---------------------
Unified turtle controller with two selectable modes:

  mode = 'trapezoid'  →  Publishes a trapezoidal (or triangular) linear
                          velocity profile for straight-line motion.

  mode = 'pid'        →  PID position controller that drives the turtle
                          to a goal position received via the topic
                          /goal_position (geometry_msgs/Point).
                          Publish a new point at any time to update the goal.

  ros2 launch trajectory_publisher turtlesim_trapezoid.launch.py mode:=pid
  ros2 launch trajectory_publisher turtlesim_trapezoid.launch.py mode:=trapezoid distance:=6.0

  # Send a goal manually:
  ros2 topic pub /goal_position geometry_msgs/Point "{x: 10.0, y: 7.5, z: 0.0}"
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from turtlesim.msg import Pose


# ── PID helper ──────────────────────────────────────────────────────────────
class PIDController:
    """Simple discrete PID with anti-windup clamp."""

    def __init__(self, kp: float, ki: float, kd: float,
                 output_min: float = -float('inf'),
                 output_max: float =  float('inf')):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self._integral   = 0.0
        self._prev_error = 0.0

    def compute(self, error: float, dt: float) -> float:
        if dt <= 0:
            return 0.0
        self._integral  += error * dt
        derivative       = (error - self._prev_error) / dt
        self._prev_error = error
        raw = self.kp * error + self.ki * self._integral + self.kd * derivative
        return float(max(self.output_min, min(self.output_max, raw)))

    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0


# ── Main node ────────────────────────────────────────────────────────────────
class TurtleController(Node):

    # State machine labels (shared by both modes)
    _WAITING  = 'WAITING'
    _ACCEL    = 'ACCELERATING'   # trapezoid only
    _CONSTANT = 'CONSTANT'       # trapezoid only
    _DECEL    = 'DECELERATING'   # trapezoid only
    _ACTIVE   = 'ACTIVE'         # pid only
    _DONE     = 'DONE'

    def __init__(self):
        super().__init__('turtle_controller')

        # ── common parameters ────────────────────────────────────────────
        self.declare_parameter('mode',        'trapezoid')   # 'trapezoid' | 'pid'
        self.declare_parameter('turtle_name', 'turtle1')
        self.declare_parameter('dt',           0.01)         # timer period [s]
        self.declare_parameter('start_delay',  2.0)          # wait before moving [s]

        # ── trapezoid parameters ─────────────────────────────────────────
        self.declare_parameter('v_max',    2.0)   # m/s
        self.declare_parameter('a_max',    0.5)   # m/s²
        self.declare_parameter('distance', 5.0)   # m

        # ── PID parameters ───────────────────────────────────────────────
        self.declare_parameter('goal_tolerance',  0.05)   # m
        self.declare_parameter('v_max_pid',       2.0)    # m/s  linear clamp
        self.declare_parameter('w_max_pid',       3.0)    # rad/s angular clamp
        self.declare_parameter('kp_linear',       1.5)
        self.declare_parameter('ki_linear',       0.0)
        self.declare_parameter('kd_linear',       0.05)
        self.declare_parameter('kp_angular',      5.0)
        self.declare_parameter('ki_angular',      0.0)
        self.declare_parameter('kd_angular',      0.1)

        # ── read common params ────────────────────────────────────────────
        self.mode        = self.get_parameter('mode').value
        self.turtle_name = self.get_parameter('turtle_name').value
        self.dt          = float(self.get_parameter('dt').value)
        self.start_delay = float(self.get_parameter('start_delay').value)

        # ── publisher ────────────────────────────────────────────────────
        self.cmd_vel_pub = self.create_publisher(
            Twist, f'{self.turtle_name}/cmd_vel', 10
        )

        # ── mode-specific init ────────────────────────────────────────────
        if self.mode == 'trapezoid':
            self._init_trapezoid()
        elif self.mode == 'pid':
            self._init_pid()
        else:
            self.get_logger().fatal(
                f"Unknown mode '{self.mode}'. Use 'trapezoid' or 'pid'."
            )
            return

        # ── runtime state ─────────────────────────────────────────────────
        self._wait  = 0.0
        self._phase = self._WAITING
        self._timer = self.create_timer(self.dt, self._timer_cb)

        self.get_logger().info(
            f'\n┌─ TurtleController ready ─────────────────────\n'
            f'│  mode        : {self.mode}\n'
            f'│  turtle      : {self.turtle_name}\n'
            f'│  start delay : {self.start_delay} s\n'
            f'└──────────────────────────────────────────────'
        )

    # ── trapezoid init ────────────────────────────────────────────────────
    def _init_trapezoid(self):
        self.v_max    = float(self.get_parameter('v_max').value)
        self.a_max    = float(self.get_parameter('a_max').value)
        self.distance = float(self.get_parameter('distance').value)
        self._t = 0.0
        self._compute_trapezoid_profile()

        self.get_logger().info(
            f'\n  [trapezoid]\n'
            f'  v_max    = {self.v_peak:.3f} m/s\n'
            f'  a_max    = {self.a_max:.3f} m/s²\n'
            f'  distance = {self.distance:.3f} m\n'
            f'  t_end    = {self.t3:.3f} s\n'
            f'  profile  = {"triangular" if self._triangular else "trapezoidal"}'
        )

    def _compute_trapezoid_profile(self):
        t_accel = self.v_max / self.a_max
        d_accel = 0.5 * self.a_max * t_accel ** 2
        if 2.0 * d_accel >= self.distance:
            self._triangular = True
            self.v_peak = math.sqrt(self.a_max * self.distance)
            t_ramp = self.v_peak / self.a_max
            self.t1 = t_ramp
            self.t2 = t_ramp
            self.t3 = 2.0 * t_ramp
        else:
            self._triangular = False
            self.v_peak = self.v_max
            d_const = self.distance - 2.0 * d_accel
            self.t1 = t_accel
            self.t2 = t_accel + d_const / self.v_max
            self.t3 = self.t2 + t_accel

    def _velocity_at(self, t: float) -> float:
        if t <= self.t1:
            return self.a_max * t
        elif t <= self.t2:
            return self.v_peak
        elif t <= self.t3:
            return self.v_peak - self.a_max * (t - self.t2)
        return 0.0

    # ── PID init ──────────────────────────────────────────────────────────
    def _init_pid(self):
        self.goal_x: float | None = None   # set by /goal_position topic
        self.goal_y: float | None = None
        self.goal_tolerance = float(self.get_parameter('goal_tolerance').value)
        v_lim = float(self.get_parameter('v_max_pid').value)
        w_lim = float(self.get_parameter('w_max_pid').value)

        self.pid_linear = PIDController(
            kp=float(self.get_parameter('kp_linear').value),
            ki=float(self.get_parameter('ki_linear').value),
            kd=float(self.get_parameter('kd_linear').value),
            output_min=-v_lim, output_max=v_lim,
        )
        self.pid_angular = PIDController(
            kp=float(self.get_parameter('kp_angular').value),
            ki=float(self.get_parameter('ki_angular').value),
            kd=float(self.get_parameter('kd_angular').value),
            output_min=-w_lim, output_max=w_lim,
        )

        self._current_pose: Pose | None = None
        self.create_subscription(
            Pose, f'{self.turtle_name}/pose', self._pose_cb, 10
        )
        self.create_subscription(
            Point, 'goal_position', self._goal_cb, 10
        )

        self.get_logger().info(
            f'\n  [pid]\n'
            f'  goal            = waiting for /goal_position topic\n'
            f'  tolerance       = {self.goal_tolerance:.3f} m\n'
            f'  linear  PID     : Kp={self.pid_linear.kp}  '
            f'Ki={self.pid_linear.ki}  Kd={self.pid_linear.kd}\n'
            f'  angular PID     : Kp={self.pid_angular.kp}  '
            f'Ki={self.pid_angular.ki}  Kd={self.pid_angular.kd}'
        )

    def _pose_cb(self, msg: Pose):
        self._current_pose = msg

    def _goal_cb(self, msg: Point):
        self.goal_x = msg.x
        self.goal_y = msg.y
        self.pid_linear.reset()
        self.pid_angular.reset()
        self._phase = self._ACTIVE
        self.get_logger().info(
            f'[PID] New goal received: ({self.goal_x:.3f}, {self.goal_y:.3f}) m'
        )

    # ── timer callback ────────────────────────────────────────────────────
    def _timer_cb(self):
        # --- wait before starting ---
        if self._phase == self._WAITING:
            self._wait += self.dt
            if self._wait >= self.start_delay:
                self._phase = self._ACCEL if self.mode == 'trapezoid' else self._ACTIVE
                self.get_logger().info(
                    f'[{self.mode.upper()}] Control started.'
                )
            return

        if self._phase == self._DONE:
            return

        if self.mode == 'trapezoid':
            self._step_trapezoid()
        else:
            self._step_pid()

    # ── trapezoid step ────────────────────────────────────────────────────
    def _step_trapezoid(self):
        v = self._velocity_at(self._t)
        self._publish(v, 0.0)

        if self._phase == self._ACCEL and self._t > self.t1:
            self._phase = self._CONSTANT if not self._triangular else self._DECEL
            self.get_logger().info(
                f'Phase → {"CONSTANT" if not self._triangular else "DECELERATING"}'
            )
        elif self._phase == self._CONSTANT and self._t > self.t2:
            self._phase = self._DECEL
            self.get_logger().info('Phase → DECELERATING')
        elif self._t > self.t3:
            self._publish(0.0, 0.0)
            self._phase = self._DONE
            self._timer.cancel()
            self.get_logger().info('Trapezoid trajectory complete.')

        self._t += self.dt

    # ── PID step ──────────────────────────────────────────────────────────
    def _step_pid(self):
        if self._current_pose is None or self.goal_x is None or self.goal_y is None:
            return

        pose = self._current_pose
        dx   = self.goal_x - pose.x
        dy   = self.goal_y - pose.y
        dist = math.sqrt(dx * dx + dy * dy)

        # --- goal reached ---
        if dist < self.goal_tolerance:
            self._publish(0.0, 0.0)
            self._phase = self._DONE
            self.get_logger().info(
                f'[PID] Goal reached!  pos=({pose.x:.3f}, {pose.y:.3f})  '
                f'error={dist:.4f} m  — waiting for next /goal_position'
            )
            return

        # --- compute heading error ---
        angle_to_goal = math.atan2(dy, dx)
        angle_error   = angle_to_goal - pose.theta
        # normalize to [-π, π]
        angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi

        # --- PID outputs ---
        v = self.pid_linear.compute(dist, self.dt)
        w = self.pid_angular.compute(angle_error, self.dt)

        # Scale down linear speed when heading is far off
        # (prevents overshooting while still turning)
        heading_factor = max(0.0, math.cos(angle_error))
        v *= heading_factor

        self._publish(v, w)

    # ── helpers ───────────────────────────────────────────────────────────
    def _publish(self, linear: float, angular: float = 0.0):
        msg = Twist()
        msg.linear.x  = float(linear)
        msg.angular.z = float(angular)
        self.cmd_vel_pub.publish(msg)


# ── entry point ──────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = TurtleController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
