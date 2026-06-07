#!/usr/bin/python3

"""
turtle_plotter.py
-----------------
Records experimental data and (optionally) renders a live matplotlib plot.

Records to ``~/comparison_logs/run_{timestamp}/`` on shutdown (Ctrl-C):

  pose.csv               turtle's actual pose over time
  cmd_vel.csv            velocity commands published to the turtle
  cfm_setpoints.csv      every CFM-generated setpoint trajectory (long format)
  classical_setpoints.csv  every classical setpoint published (long format)
  goals.csv              every /traj_context message received
  plot.png               final matplotlib snapshot (only if live_plot=true)

Parameters:
  turtle_name   default "turtle1"
  live_plot     bool, default true   — open matplotlib window for live feedback
  plot_rate_hz  default 5.0          — redraw rate
  max_points    default 2000         — live-plot rolling buffer (CSVs are unbounded)
  run_dir       optional explicit path; default auto-generated

Offline plotting:
  python3 plot_run.py ~/comparison_logs/run_<timestamp>
"""

import csv
import json
import os
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseArray, Point
from std_msgs.msg import Float32
from turtlesim.msg import Pose

try:
    from llm_pack_interface.msg import TrajContext   # type: ignore
    _HAS_TRAJ_CONTEXT = True
except ImportError:
    TrajContext = None                                # type: ignore
    _HAS_TRAJ_CONTEXT = False

# matplotlib imports are deferred so live_plot=false doesn't even import TkAgg


class TurtlePlotter(Node):

    def __init__(self):
        super().__init__('turtle_plotter')

        self.declare_parameter('turtle_name',  'turtle1')
        self.declare_parameter('live_plot',    True)
        self.declare_parameter('plot_rate_hz', 5.0)
        self.declare_parameter('max_points',   2000)
        self.declare_parameter('run_dir',      '')   # if empty, auto-generated

        self.turtle_name = self.get_parameter('turtle_name').value
        self.live_plot   = bool(self.get_parameter('live_plot').value)
        self.max_pts     = int(self.get_parameter('max_points').value)
        self._plot_period = 1.0 / float(self.get_parameter('plot_rate_hz').value)

        # ---- output directory --------------------------------------------
        run_dir = self.get_parameter('run_dir').value
        if not run_dir:
            stamp = time.strftime('%Y%m%d-%H%M%S')
            run_dir = os.path.expanduser(f'~/comparison_logs/run_{stamp}')
        self._run_dir = run_dir
        os.makedirs(self._run_dir, exist_ok=True)

        # ---- data buffers (also used for live plot) ----------------------
        self._t_pose  : list[float] = []
        self._x       : list[float] = []
        self._y       : list[float] = []
        self._theta   : list[float] = []
        self._v_actual: list[float] = []
        self._w_actual: list[float] = []

        self._t_cmd   : list[float] = []
        self._v_cmd   : list[float] = []
        self._w_cmd   : list[float] = []

        # latest setpoint trajectories (for live plot overlay)
        self._cfm_setpoint_xy:       list[tuple[float, float]] = []
        self._classical_setpoint_xy: list[tuple[float, float]] = []

        # ---- streaming CSV writers (open for append; flushed on each write) ----
        self._csv = {
            'pose':       self._open_csv('pose.csv',
                          ['time', 'x', 'y', 'theta', 'linear_velocity', 'angular_velocity']),
            'cmd_vel':    self._open_csv('cmd_vel.csv',
                          ['time', 'linear_x', 'angular_z']),
            # resample_idx: sequential index of this CFM plan (0=initial, 1=first replan, ...)
            'cfm':        self._open_csv('cfm_setpoints.csv',
                          ['publish_time', 'resample_idx', 'wp_idx', 'x', 'y', 'theta']),
            'classical':  self._open_csv('classical_setpoints.csv',
                          ['publish_time', 'wp_idx', 'x', 'y', 'theta']),
            # goal_x_body/y_body = raw LLM local-frame output
            # goal_x_world/y_world = after body→world conversion in controller
            'goals':      self._open_csv('goals.csv',
                          ['time', 'goal_x_body', 'goal_y_body', 'goal_z_body',
                           'v_const', 'a', 'part',
                           'goal_x_world', 'goal_y_world']),
            # CFM trajectory-generation latency per sample (ms) — no log parsing needed
            'cfm_gen':    self._open_csv('cfm_gen_times.csv',
                          ['time', 'gen_ms']),
        }
        self._cfm_publish_count       = 0
        self._classical_publish_count = 0
        self._goal_count              = 0
        self._cfm_gen_count           = 0

        # Cache latest world-frame goal published by controller (body→world converted)
        self._latest_goal_world_x: float | None = None
        self._latest_goal_world_y: float | None = None

        self._t0: float | None = None

        # ---- subscriptions ----------------------------------------------
        self.create_subscription(Pose,    f'{self.turtle_name}/pose',
                                 self._pose_cb, 10)
        self.create_subscription(Twist,   f'{self.turtle_name}/cmd_vel',
                                 self._cmd_cb, 10)
        self.create_subscription(PoseArray, 'cfm_setpoint',
                                 self._cfm_setpoint_cb, 10)
        self.create_subscription(PoseArray, 'classical_setpoint',
                                 self._classical_setpoint_cb, 10)
        self.create_subscription(Point, 'goal_world_pos',
                                 self._goal_world_cb, 10)
        self.create_subscription(Float32, 'cfm_gen_ms',
                                 self._cfm_gen_cb, 10)
        if _HAS_TRAJ_CONTEXT:
            self.create_subscription(TrajContext, 'traj_context',
                                     self._goal_cb, 10)

        # ---- optional matplotlib setup ----------------------------------
        self._fig = self._axes = None
        if self.live_plot:
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            self._plt = plt
            plt.ion()
            self._fig, self._axes = plt.subplots(2, 2, figsize=(13, 8))
            self._fig.suptitle(
                f'CFM vs Classical Trapezoid  –  turtle: {self.turtle_name}',
                fontsize=13, fontweight='bold')
            self._fig.tight_layout(pad=3.0)
            plt.pause(0.01)
        else:
            self._plt = None

        # ---- run-level metadata -----------------------------------------
        meta = {
            'run_dir':    self._run_dir,
            'start_time': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'turtle_name': self.turtle_name,
            'csv_schema': {
                'pose.csv':               'time, x, y, theta, linear_velocity, angular_velocity',
                'cmd_vel.csv':            'time, linear_x, angular_z',
                'cfm_setpoints.csv':      'publish_time, resample_idx, wp_idx, x, y, theta',
                'classical_setpoints.csv':'publish_time, wp_idx, x, y, theta',
                'goals.csv':              'time, goal_x_body, goal_y_body, goal_z_body, v_const, a, part, goal_x_world, goal_y_world',
            },
        }
        with open(os.path.join(self._run_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        self.get_logger().info(
            f'Plotter ready. Recording → {self._run_dir} '
            f'(live_plot={self.live_plot})')

    # ── helpers ───────────────────────────────────────────────────────────
    def _now(self) -> float:
        t = time.monotonic()
        if self._t0 is None:
            self._t0 = t
        return t - self._t0

    def _trim(self, lst: list) -> list:
        return lst[-self.max_pts:]

    def _open_csv(self, name: str, header: list[str]):
        path = os.path.join(self._run_dir, name)
        f = open(path, 'w', newline='')
        w = csv.writer(f)
        w.writerow(header)
        return {'file': f, 'writer': w}

    def _csv_row(self, key: str, row: list):
        ch = self._csv[key]
        ch['writer'].writerow(row)
        ch['file'].flush()

    # ── subscribers ───────────────────────────────────────────────────────
    def _pose_cb(self, msg: Pose):
        t = self._now()
        self._t_pose  = self._trim(self._t_pose + [t])
        self._x       = self._trim(self._x + [msg.x])
        self._y       = self._trim(self._y + [msg.y])
        self._theta   = self._trim(self._theta + [msg.theta])
        self._v_actual= self._trim(self._v_actual + [msg.linear_velocity])
        self._w_actual= self._trim(self._w_actual + [msg.angular_velocity])
        self._csv_row('pose', [f'{t:.4f}',
                               f'{msg.x:.4f}', f'{msg.y:.4f}', f'{msg.theta:.4f}',
                               f'{msg.linear_velocity:.4f}',
                               f'{msg.angular_velocity:.4f}'])

    def _cmd_cb(self, msg: Twist):
        t = self._now()
        self._t_cmd = self._trim(self._t_cmd + [t])
        self._v_cmd = self._trim(self._v_cmd + [msg.linear.x])
        self._w_cmd = self._trim(self._w_cmd + [msg.angular.z])
        self._csv_row('cmd_vel', [f'{t:.4f}',
                                  f'{msg.linear.x:.4f}',
                                  f'{msg.angular.z:.4f}'])

    def _cfm_setpoint_cb(self, msg: PoseArray):
        t = self._now()
        resample_idx = self._cfm_publish_count   # 0-based sequential replan index
        self._cfm_publish_count += 1
        # update live overlay
        self._cfm_setpoint_xy = [(p.position.x, p.position.y) for p in msg.poses]
        # CSV: long format, one row per waypoint
        for i, p in enumerate(msg.poses):
            import math as _m
            th = 2.0 * _m.atan2(p.orientation.z, p.orientation.w)
            self._csv_row('cfm', [f'{t:.4f}', resample_idx, i,
                                  f'{p.position.x:.4f}',
                                  f'{p.position.y:.4f}',
                                  f'{th:.4f}'])

    def _classical_setpoint_cb(self, msg: PoseArray):
        t = self._now()
        self._classical_publish_count += 1
        self._classical_setpoint_xy = [(p.position.x, p.position.y) for p in msg.poses]
        for i, p in enumerate(msg.poses):
            import math as _m
            th = 2.0 * _m.atan2(p.orientation.z, p.orientation.w)
            self._csv_row('classical', [f'{t:.4f}', i,
                                        f'{p.position.x:.4f}',
                                        f'{p.position.y:.4f}',
                                        f'{th:.4f}'])

    def _goal_world_cb(self, msg: Point):
        """Cache world-frame goal published by controller (arrives just before /traj_context)."""
        self._latest_goal_world_x = msg.x
        self._latest_goal_world_y = msg.y

    def _cfm_gen_cb(self, msg: Float32):
        """Record CFM trajectory-generation latency (ms) per sample."""
        t = self._now()
        self._cfm_gen_count += 1
        self._csv_row('cfm_gen', [f'{t:.4f}', f'{float(msg.data):.3f}'])

    def _goal_cb(self, msg):
        t = self._now()
        self._goal_count += 1
        gx_w = self._latest_goal_world_x if self._latest_goal_world_x is not None else float('nan')
        gy_w = self._latest_goal_world_y if self._latest_goal_world_y is not None else float('nan')
        self._csv_row('goals', [f'{t:.4f}',
                                f'{msg.s_goal.x:.4f}',   # LLM body-frame output
                                f'{msg.s_goal.y:.4f}',
                                f'{msg.s_goal.z:.4f}',
                                f'{msg.v_const:.4f}',
                                f'{msg.a:.4f}',
                                int(msg.part),
                                f'{gx_w:.4f}',            # world-frame (after body→world)
                                f'{gy_w:.4f}'])

    # ── live plot ─────────────────────────────────────────────────────────
    def update_plot(self):
        if not self.live_plot or len(self._t_pose) < 2:
            return

        plt = self._plt
        ax_pos, ax_vel, ax_xy, ax_theta = (
            self._axes[0, 0], self._axes[0, 1],
            self._axes[1, 0], self._axes[1, 1]
        )
        for ax in self._axes.flatten():
            ax.cla()

        tp = self._t_pose

        ax_pos.plot(tp, self._x, 'b-', linewidth=1.5, label='x')
        ax_pos.plot(tp, self._y, 'r-', linewidth=1.5, label='y')
        ax_pos.set_xlabel('Time (s)'); ax_pos.set_ylabel('Position (m)')
        ax_pos.set_title('Position vs Time')
        ax_pos.legend(loc='upper left'); ax_pos.grid(True, alpha=0.4)

        ax_vel.plot(tp, self._v_actual, 'b-', linewidth=1.5, label='Actual (pose)')
        if len(self._t_cmd) >= 2:
            ax_vel.plot(self._t_cmd, self._v_cmd, 'r--', linewidth=1.5, label='Commanded')
        ax_vel.set_xlabel('Time (s)'); ax_vel.set_ylabel('Linear Velocity (m/s)')
        ax_vel.set_title('Linear Velocity vs Time')
        ax_vel.legend(loc='upper right'); ax_vel.grid(True, alpha=0.4)

        if self._classical_setpoint_xy:
            cx, cy = zip(*self._classical_setpoint_xy)
            ax_xy.plot(cx, cy, color='orange', linestyle=':', linewidth=2.0,
                       alpha=0.85, label='Classical setpoint', zorder=2)
        if self._cfm_setpoint_xy:
            fx, fy = zip(*self._cfm_setpoint_xy)
            ax_xy.plot(fx, fy, 'b--', linewidth=1.6, alpha=0.85,
                       label='CFM setpoint', zorder=3)
        ax_xy.plot(self._x, self._y, 'g-', linewidth=1.6,
                   label='Turtle actual', zorder=4)
        ax_xy.scatter(self._x[0], self._y[0], color='green',
                      zorder=5, s=60, label='Start')
        ax_xy.scatter(self._x[-1], self._y[-1], color='red',
                      zorder=5, s=60, label='Current')
        ax_xy.set_xlabel('X (m)'); ax_xy.set_ylabel('Y (m)')
        ax_xy.set_title('XY Trajectory — Setpoints vs Actual')
        ax_xy.legend(loc='best', fontsize=8)
        ax_xy.grid(True, alpha=0.4)
        ax_xy.set_aspect('equal', adjustable='datalim')

        ax_theta.plot(tp, self._theta, color='purple',
                      linewidth=1.5, label='θ (rad)')
        ax_theta.set_xlabel('Time (s)'); ax_theta.set_ylabel('Theta (rad)')
        ax_theta.set_title('Heading vs Time')
        ax_theta.legend(loc='upper right'); ax_theta.grid(True, alpha=0.4)

        self._fig.tight_layout(pad=3.0)
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    # ── shutdown ──────────────────────────────────────────────────────────
    def save_plot(self):
        if self.live_plot and self._fig is not None:
            save_path = os.path.join(self._run_dir, 'plot.png')
            self._fig.savefig(save_path, dpi=150, bbox_inches='tight')
            self.get_logger().info(f'Plot saved → {save_path}')

    def close_csvs(self):
        for ch in self._csv.values():
            try:
                ch['file'].close()
            except Exception:
                pass
        self.get_logger().info(
            f'Recorded → {self._run_dir} '
            f'(pose rows={len(self._t_pose)}, cmd_vel rows={len(self._t_cmd)}, '
            f'cfm publishes={self._cfm_publish_count}, '
            f'classical publishes={self._classical_publish_count}, '
            f'goals={self._goal_count})'
        )


# -----------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = TurtlePlotter()

    plot_period = node._plot_period
    last_plot   = 0.0

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            if node.live_plot:
                now = time.monotonic()
                if now - last_plot >= plot_period:
                    node.update_plot()
                    last_plot = now
    except KeyboardInterrupt:
        pass
    finally:
        if node.live_plot:
            node.update_plot()
            node.save_plot()
            if node._plt is not None:
                node._plt.ioff()
                node._plt.show(block=False)
                node._plt.pause(2)
        node.close_csvs()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
