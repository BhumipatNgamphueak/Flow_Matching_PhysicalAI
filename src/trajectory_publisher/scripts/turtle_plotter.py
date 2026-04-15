#!/usr/bin/python3

"""
turtle_plotter.py
-----------------
Subscribes to a turtle's /pose and /cmd_vel topics and produces a
4-panel real-time plot:

  [0,0]  Position  (x, y)  vs time
  [0,1]  Linear velocity   – actual (from pose) vs commanded (from cmd_vel)
  [1,0]  XY trajectory path
  [1,1]  Heading  (theta)  vs time

Press Ctrl-C to stop; the final plot is saved to ~/turtle_trajectory.png.
"""

import os
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose

import matplotlib
matplotlib.use('TkAgg')          # works in most desktop Linux setups
import matplotlib.pyplot as plt


class TurtlePlotter(Node):

    def __init__(self):
        super().__init__('turtle_plotter')

        self.declare_parameter('turtle_name',  'turtle1')
        self.declare_parameter('plot_rate_hz', 5.0)       # redraw rate
        self.declare_parameter('max_points',   2000)       # rolling buffer size

        self.turtle_name  = self.get_parameter('turtle_name').value
        self.max_pts      = int(self.get_parameter('max_points').value)
        self._plot_period = 1.0 / float(self.get_parameter('plot_rate_hz').value)

        # ---- data buffers ------------------------------------------------
        self._t_pose  : list[float] = []   # timestamp from /pose
        self._x       : list[float] = []
        self._y       : list[float] = []
        self._theta   : list[float] = []
        self._v_actual: list[float] = []   # linear_velocity in Pose msg

        self._t_cmd   : list[float] = []   # timestamp from /cmd_vel
        self._v_cmd   : list[float] = []   # commanded linear.x

        self._t0: float | None = None       # wall-clock start time

        # ---- subscriptions -----------------------------------------------
        self.create_subscription(
            Pose,
            f'{self.turtle_name}/pose',
            self._pose_cb,
            10
        )
        self.create_subscription(
            Twist,
            f'{self.turtle_name}/cmd_vel',
            self._cmd_cb,
            10
        )

        # ---- matplotlib setup --------------------------------------------
        plt.ion()
        self._fig, self._axes = plt.subplots(2, 2, figsize=(13, 8))
        self._fig.suptitle(
            f'Trapezoidal Velocity Profile  –  turtle: {self.turtle_name}',
            fontsize=13, fontweight='bold'
        )
        self._fig.tight_layout(pad=3.0)
        plt.pause(0.01)

        self.get_logger().info(
            f'Plotter ready.  Subscribing to '
            f'/{self.turtle_name}/pose  and  /{self.turtle_name}/cmd_vel'
        )

    # ------------------------------------------------------------------
    def _now(self) -> float:
        """Wall-clock seconds since the first received message."""
        t = time.monotonic()
        if self._t0 is None:
            self._t0 = t
        return t - self._t0

    def _trim(self, lst: list) -> list:
        return lst[-self.max_pts:]

    # ------------------------------------------------------------------
    def _pose_cb(self, msg: Pose):
        t = self._now()
        self._t_pose  = self._trim(self._t_pose  + [t])
        self._x       = self._trim(self._x       + [msg.x])
        self._y       = self._trim(self._y       + [msg.y])
        self._theta   = self._trim(self._theta   + [msg.theta])
        self._v_actual= self._trim(self._v_actual+ [msg.linear_velocity])

    def _cmd_cb(self, msg: Twist):
        t = self._now()
        self._t_cmd = self._trim(self._t_cmd + [t])
        self._v_cmd = self._trim(self._v_cmd + [msg.linear.x])

    # ------------------------------------------------------------------
    def update_plot(self):
        """Redraw all four axes with the latest buffered data."""
        if len(self._t_pose) < 2:
            return

        ax_pos, ax_vel, ax_xy, ax_theta = (
            self._axes[0, 0], self._axes[0, 1],
            self._axes[1, 0], self._axes[1, 1]
        )

        for ax in self._axes.flatten():
            ax.cla()

        tp = self._t_pose

        # --- [0,0]  Position vs time --------------------------------
        ax_pos.plot(tp, self._x,     'b-', linewidth=1.5, label='x')
        ax_pos.plot(tp, self._y,     'r-', linewidth=1.5, label='y')
        ax_pos.set_xlabel('Time (s)')
        ax_pos.set_ylabel('Position (m)')
        ax_pos.set_title('Position vs Time')
        ax_pos.legend(loc='upper left')
        ax_pos.grid(True, alpha=0.4)

        # --- [0,1]  Velocity vs time --------------------------------
        ax_vel.plot(tp, self._v_actual,
                    'b-', linewidth=1.5, label='Actual (pose)')
        if len(self._t_cmd) >= 2:
            ax_vel.plot(self._t_cmd, self._v_cmd,
                        'r--', linewidth=1.5, label='Commanded')
        ax_vel.set_xlabel('Time (s)')
        ax_vel.set_ylabel('Linear Velocity (m/s)')
        ax_vel.set_title('Linear Velocity vs Time')
        ax_vel.legend(loc='upper right')
        ax_vel.grid(True, alpha=0.4)

        # --- [1,0]  XY trajectory -----------------------------------
        ax_xy.plot(self._x, self._y, 'g-', linewidth=1.5)
        ax_xy.scatter(self._x[0],  self._y[0],
                      color='green', zorder=5, s=60, label='Start')
        ax_xy.scatter(self._x[-1], self._y[-1],
                      color='red',   zorder=5, s=60, label='End')
        ax_xy.set_xlabel('X (m)')
        ax_xy.set_ylabel('Y (m)')
        ax_xy.set_title('XY Trajectory')
        ax_xy.legend(loc='upper left')
        ax_xy.grid(True, alpha=0.4)
        ax_xy.set_aspect('equal', adjustable='datalim')

        # --- [1,1]  Heading vs time ---------------------------------
        ax_theta.plot(tp, self._theta, color='purple',
                      linewidth=1.5, label='θ (rad)')
        ax_theta.set_xlabel('Time (s)')
        ax_theta.set_ylabel('Theta (rad)')
        ax_theta.set_title('Heading vs Time')
        ax_theta.legend(loc='upper right')
        ax_theta.grid(True, alpha=0.4)

        self._fig.tight_layout(pad=3.0)
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    # ------------------------------------------------------------------
    def save_plot(self):
        save_path = os.path.expanduser('~/turtle_trajectory.png')
        self._fig.savefig(save_path, dpi=150, bbox_inches='tight')
        self.get_logger().info(f'Plot saved → {save_path}')


# -----------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = TurtlePlotter()

    plot_period = node._plot_period
    last_plot   = 0.0

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            now = time.monotonic()
            if now - last_plot >= plot_period:
                node.update_plot()
                last_plot = now
    except KeyboardInterrupt:
        pass
    finally:
        node.update_plot()   # final render
        node.save_plot()
        plt.ioff()
        plt.show(block=False)
        plt.pause(2)         # keep window open briefly before exit

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
