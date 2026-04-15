#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import math


class TrapezoidVelocityPublisher(Node):
    """
    Publishes a trapezoidal linear velocity profile to a turtle's /cmd_vel topic.

    Profile phases:
        Acceleration : 0 → v_max  at rate  a_max
        Constant     : v_max for the middle segment
        Deceleration : v_max → 0  at rate -a_max

    If the requested distance is too short to reach v_max, a triangular
    (bang-bang) profile is used instead.
    """

    # Internal state machine
    _WAITING    = 'WAITING'
    _ACCEL      = 'ACCELERATING'
    _CONSTANT   = 'CONSTANT'
    _DECEL      = 'DECELERATING'
    _DONE       = 'DONE'

    def __init__(self):
        super().__init__('trapezoid_velocity_publisher')

        # ---------- parameters ----------
        self.declare_parameter('v_max',       2.0)   # m/s
        self.declare_parameter('a_max',       0.5)   # m/s²
        self.declare_parameter('distance',    3.0)   # m  (straight-line)
        self.declare_parameter('turtle_name', 'turtle1')
        self.declare_parameter('start_delay', 2.0)   # s  (wait before moving)
        self.declare_parameter('dt',          0.01)  # s  (timer period)

        self.v_max       = float(self.get_parameter('v_max').value)
        self.a_max       = float(self.get_parameter('a_max').value)
        self.distance    = float(self.get_parameter('distance').value)
        self.turtle_name = self.get_parameter('turtle_name').value
        self.start_delay = float(self.get_parameter('start_delay').value)
        self.dt          = float(self.get_parameter('dt').value)

        self._compute_profile()

        # ---------- ROS interfaces ----------
        self.cmd_vel_pub = self.create_publisher(
            Twist, f'{self.turtle_name}/cmd_vel', 10
        )

        # ---------- runtime state ----------
        self._t       = 0.0          # elapsed trajectory time (s)
        self._wait    = 0.0          # elapsed waiting time (s)
        self._phase   = self._WAITING
        self._timer   = self.create_timer(self.dt, self._timer_cb)

        self.get_logger().info(
            f'Trapezoidal publisher ready.\n'
            f'  turtle : {self.turtle_name}\n'
            f'  v_max  : {self.v_peak:.3f} m/s\n'
            f'  a_max  : {self.a_max:.3f} m/s²\n'
            f'  dist   : {self.distance:.3f} m\n'
            f'  t_end  : {self.t3:.3f} s\n'
            f'  profile: {"triangular" if self._triangular else "trapezoidal"}\n'
            f'  delay  : {self.start_delay:.1f} s'
        )

    # ------------------------------------------------------------------
    def _compute_profile(self):
        """Pre-compute the three phase boundary times."""
        t_accel  = self.v_max / self.a_max
        d_accel  = 0.5 * self.a_max * t_accel ** 2   # distance per ramp

        if 2.0 * d_accel >= self.distance:
            # Triangular profile – peak velocity is lower than v_max
            self._triangular = True
            self.v_peak = math.sqrt(self.a_max * self.distance)
            t_ramp      = self.v_peak / self.a_max
            self.t1 = t_ramp          # end of acceleration
            self.t2 = t_ramp          # zero-length constant phase
            self.t3 = 2.0 * t_ramp   # end of deceleration
        else:
            self._triangular = False
            self.v_peak = self.v_max
            d_const = self.distance - 2.0 * d_accel
            self.t1 = t_accel
            self.t2 = t_accel + d_const / self.v_max
            self.t3 = self.t2 + t_accel

    # ------------------------------------------------------------------
    def _velocity_at(self, t: float) -> float:
        """Return the commanded linear velocity at trajectory time t."""
        if t < 0.0:
            return 0.0
        elif t <= self.t1:
            return self.a_max * t
        elif t <= self.t2:
            return self.v_peak
        elif t <= self.t3:
            return self.v_peak - self.a_max * (t - self.t2)
        else:
            return 0.0

    # ------------------------------------------------------------------
    def _publish_twist(self, linear: float, angular: float = 0.0):
        msg = Twist()
        msg.linear.x  = linear
        msg.angular.z = angular
        self.cmd_vel_pub.publish(msg)

    # ------------------------------------------------------------------
    def _timer_cb(self):
        # --- waiting phase ---
        if self._phase == self._WAITING:
            self._wait += self.dt
            if self._wait >= self.start_delay:
                self._phase = self._ACCEL
                self.get_logger().info('Trajectory started – ACCELERATING')
            return

        # --- done phase ---
        if self._phase == self._DONE:
            return

        # --- active trajectory ---
        v = self._velocity_at(self._t)
        self._publish_twist(v)

        # Phase transition logging
        if self._phase == self._ACCEL and self._t > self.t1:
            self._phase = self._CONSTANT if not self._triangular else self._DECEL
            self.get_logger().info(
                f'Phase → {"CONSTANT" if not self._triangular else "DECELERATING"}'
            )
        elif self._phase == self._CONSTANT and self._t > self.t2:
            self._phase = self._DECEL
            self.get_logger().info('Phase → DECELERATING')
        elif self._t > self.t3:
            self._publish_twist(0.0)
            self._phase = self._DONE
            self._timer.cancel()
            self.get_logger().info('Trajectory complete – velocity zeroed.')

        self._t += self.dt


# -----------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = TrapezoidVelocityPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
