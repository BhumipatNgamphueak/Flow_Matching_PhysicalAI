#!/usr/bin/python3
# turtle_controller.py — LLM-driven trapezoid (classical) vs CFM (flow matching) comparison

"""
turtle_controller.py
---------------------
Unified turtle controller with two selectable modes that BOTH consume the same
LLM-issued constraints from `/traj_context` (llm_pack_interface/TrajContext):

  mode = 'trapezoid'  →  Classical trapezoidal velocity profile, computed
                          analytically from v_const / a / distance.

  mode = 'pid'        →  Conditional Flow Matching (CFM) trajectory + PID tracker.
                          Loads a trained CFM model and samples a (50, 3) trajectory
                          conditioned on the LLM constraints.

Both modes also accept manual goals on /goal_position (geometry_msgs/Point) for
backward compatibility. Launch parameters (v_max, a_max, distance, goal_x, goal_y)
act as fallback defaults until a TrajContext arrives.

Comparison logging: every executed trajectory (CFM and the analytical classical
profile recomputed from the same constraints) is saved to
  ~/comparison_logs/{timestamp}_{mode}.npz
for offline plotting.

Usage:
  ros2 launch trajectory_publisher turtlesim_trapezoid.launch.py mode:=trapezoid
  ros2 launch trajectory_publisher turtlesim_trapezoid.launch.py mode:=pid
  ros2 service call /LlmPrompt llm_pack_interface/srv/String "{prompt: 'go to x=2.0, y=1.5 with v=0.15 and a=0.03'}"
"""

import json
import math
import os
import time
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point, PoseArray, Pose as GeomPose
from std_msgs.msg import Float32
from turtlesim.msg import Pose

# Optional LLM constraint message — controller works without it (uses launch params)
try:
    from llm_pack_interface.msg import TrajContext  # type: ignore
    _HAS_TRAJ_CONTEXT = True
except ImportError:
    TrajContext = None  # type: ignore
    _HAS_TRAJ_CONTEXT = False

_SCRIPTS_DIR = os.path.dirname(os.path.realpath(__file__))
CHECKPOINT_PATH = os.path.join(
    _SCRIPTS_DIR, "logs", "pose_trajectory_3DV2",
    "cfm", "H64_T100", "20260505-1157", "state_192000.pt"
)
DEVICE           = "cpu"  # cuDNN fails to init inside ROS subprocess; CPU is fast enough (~10ms)
HORIZON          = 64            # padded horizon (must match training)
ORIGINAL_LEN     = 50            # original trajectory length
N_SAMPLING_STEPS = 10            # Euler sampling steps for CFM
CFM_DT           = 0.04          # seconds per waypoint (25 Hz playback)

# Hardcoded angular limits (training distribution; LLM only emits linear v_const/a)
OMEGA_CONST_DEFAULT = 0.5        # rad/s
ALPHA_CONST_DEFAULT = 0.05       # rad/s²

# Fallback linear limits if no TrajContext arrives (matches training distribution mid-range)
V_CONST_FALLBACK = 0.156         # m/s
A_FALLBACK       = 0.028         # m/s²

LOG_DIR = os.path.expanduser("~/comparison_logs")


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

        # ── trapezoid (classical) parameters — used as fallback ──────────
        self.declare_parameter('v_max',    2.0)   # m/s
        self.declare_parameter('a_max',    0.5)   # m/s²
        self.declare_parameter('distance', 5.0)   # m

        # ── PID parameters ───────────────────────────────────────────────
        self.declare_parameter('goal_tolerance',  0.05)   # m
        self.declare_parameter('v_max_pid',       0.156)  # m/s linear clamp
        self.declare_parameter('w_max_pid',       3.0)    # rad/s angular clamp
        self.declare_parameter('plan_once',       False)  # disable 1 Hz CFM replanning
        self.declare_parameter('kp_linear',       1.5)
        self.declare_parameter('ki_linear',       0.0)
        self.declare_parameter('kd_linear',       0.05)
        self.declare_parameter('kp_angular',      5.0)
        self.declare_parameter('ki_angular',      0.0)
        self.declare_parameter('kd_angular',      0.1)
        self.declare_parameter('angle_tolerance',    0.05)   # rad (reserved for future use)

        # ── PID/trapezoid fallback goal (used only if no TrajContext / /goal_position) ──
        self.declare_parameter('goal_x', 10.0)    # m (TurtleSim world)
        self.declare_parameter('goal_y',  7.5)    # m (TurtleSim world)

        # ── read common params ────────────────────────────────────────────
        self.mode        = self.get_parameter('mode').value
        self.turtle_name = self.get_parameter('turtle_name').value
        self.dt          = float(self.get_parameter('dt').value)
        self.start_delay = float(self.get_parameter('start_delay').value)

        # ── publishers ───────────────────────────────────────────────────
        self.cmd_vel_pub = self.create_publisher(
            Twist, f'{self.turtle_name}/cmd_vel', 10
        )
        # Setpoint trajectories for the plotter — let user compare CFM-generated
        # waypoints, the analytical classical baseline, and the turtle's actual path.
        self._cfm_path_pub       = self.create_publisher(PoseArray, 'cfm_setpoint', 10)
        self._classical_path_pub = self.create_publisher(PoseArray, 'classical_setpoint', 10)
        # CFM trajectory-generation latency (ms) per sample, for offline timing stats
        self._cfm_gen_ms_pub     = self.create_publisher(Float32, 'cfm_gen_ms', 10)
        # World-frame goal after body→world conversion (plotter subscribes for correct logging)
        self._goal_world_pub = self.create_publisher(Point, 'goal_world_pos', 10)

        # ── shared state: current pose + latest LLM constraints ───────────
        self._current_pose: Pose | None = None
        self.create_subscription(
            Pose, f'{self.turtle_name}/pose', self._pose_cb, 10
        )

        # Cache: (v_const, a, goal_x_world, goal_y_world) — None until first TrajContext
        self._llm_ctx: dict | None = None
        if _HAS_TRAJ_CONTEXT:
            self.create_subscription(
                TrajContext, 'traj_context', self._traj_context_cb, 10
            )
            self.get_logger().info('Subscribed to /traj_context (LLM constraints)')
        else:
            self.get_logger().warning(
                'llm_pack_interface not available; /traj_context disabled. '
                'Falling back to launch parameters only.'
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

        os.makedirs(LOG_DIR, exist_ok=True)
        meta = {
            'session_start': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'mode':          self.mode,
            'turtle_name':   self.turtle_name,
            'checkpoint':    CHECKPOINT_PATH if self.mode == 'pid' else None,
            'horizon':       HORIZON,
            'original_len':  ORIGINAL_LEN,
            'cfm_dt':        CFM_DT,
            'n_sampling_steps': N_SAMPLING_STEPS,
            'context_13dim_schema': [
                'goal_x_local', 'goal_y_local', 's_goal_theta',
                'v_const', 'a', 'omega_const', 'alpha_const',
                'q_x', 'q_y', 'q_theta',
                'qdot_x', 'qdot_y', 'qdot_theta',
            ],
        }
        meta_path = os.path.join(LOG_DIR, f'meta_{time.strftime("%Y%m%d-%H%M%S")}.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        self.get_logger().info(f'[meta] Session metadata → {meta_path}')

        # ── runtime state ─────────────────────────────────────────────────
        self._wait  = 0.0
        self._phase = self._WAITING
        self._timer = self.create_timer(self.dt, self._timer_cb)

        self.get_logger().info(
            f'\n┌─ TurtleController ready ─────────────────────\n'
            f'│  mode        : {self.mode}\n'
            f'│  turtle      : {self.turtle_name}\n'
            f'│  start delay : {self.start_delay} s\n'
            f'│  log dir     : {LOG_DIR}\n'
            f'└──────────────────────────────────────────────'
        )

    # ── pose subscriber ───────────────────────────────────────────────────
    def _pose_cb(self, msg: Pose):
        self._current_pose = msg

    # ── /traj_context subscriber (BOTH modes) ─────────────────────────────
    def _traj_context_cb(self, msg):
        # LLM is instructed to output goals in the robot's LOCAL body frame
        # (robot at origin, front = +x, left = +y). Convert to world frame by
        # adding current turtle world position (axes are world-aligned because
        # the turtle spawns facing +x and the CFM model uses translation-only
        # local frames, not rotated body frames).
        goal_x_local = float(msg.s_goal.x)
        goal_y_local = float(msg.s_goal.y)
        v_const      = float(msg.v_const)
        a            = float(msg.a)

        if self._current_pose is not None:
            pose_wx = self._current_pose.x
            pose_wy = self._current_pose.y
            theta   = self._current_pose.theta
        else:
            # Pose not yet received — use TurtleSim default spawn
            pose_wx, pose_wy, theta = 7.5, 7.5, 0.0
            self.get_logger().warning(
                '[TrajContext] Pose not yet available; using default spawn (7.5, 7.5, 0) '
                'for body→world conversion. Will be corrected on next resample.'
            )

        # Rotate body-frame goal into world-aligned frame, then translate
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        goal_x_world = pose_wx + goal_x_local * cos_t - goal_y_local * sin_t
        goal_y_world = pose_wy + goal_x_local * sin_t + goal_y_local * cos_t

        self._llm_ctx = {
            'v_const':      v_const,
            'a':            a,
            'goal_x_world': goal_x_world,
            'goal_y_world': goal_y_world,
            'context_wx':   pose_wx,   # pose snapshot at context-receipt time
            'context_wy':   pose_wy,   # used as classical baseline start
        }
        # Publish world-frame goal so plotter can log it correctly
        gw_msg = Point()
        gw_msg.x = goal_x_world
        gw_msg.y = goal_y_world
        gw_msg.z = 0.0
        self._goal_world_pub.publish(gw_msg)
        self.get_logger().info(
            f'[TrajContext] v_const={v_const:.4f} a={a:.4f} '
            f'goal_body=({goal_x_local:.3f}, {goal_y_local:.3f}) '
            f'→ goal_world=({goal_x_world:.3f}, {goal_y_world:.3f})'
        )

        if self.mode == 'trapezoid':
            self._apply_trapezoid_context()
        else:  # 'pid'
            self._apply_pid_context()

    # ── trapezoid init ────────────────────────────────────────────────────
    def _init_trapezoid(self):
        # Fallback constraints from launch params; will be overridden by TrajContext
        self.v_max    = float(self.get_parameter('v_max').value)
        self.a_max    = float(self.get_parameter('a_max').value)
        self.distance = float(self.get_parameter('distance').value)
        self._t       = 0.0
        # Goal in world coords (for analytical trajectory logging)
        self._goal_x_world = float(self.get_parameter('goal_x').value)
        self._goal_y_world = float(self.get_parameter('goal_y').value)
        # Start pose captured when execution begins (for kinematic integration)
        self._start_pose: tuple[float, float, float] | None = None
        self._compute_trapezoid_profile()

        self.get_logger().info(
            f'\n  [trapezoid]\n'
            f'  v_max    = {self.v_peak:.3f} m/s (fallback)\n'
            f'  a_max    = {self.a_max:.3f} m/s²  (fallback)\n'
            f'  distance = {self.distance:.3f} m  (fallback)\n'
            f'  t_end    = {self.t3:.3f} s\n'
            f'  profile  = {"triangular" if self._triangular else "trapezoidal"}'
        )

    def _apply_trapezoid_context(self):
        """Recompute trapezoid profile from latest LLM constraints + current pose."""
        if self._llm_ctx is None:
            return
        if self._current_pose is None:
            self.get_logger().warning(
                '[trapezoid] TrajContext arrived before pose; deferring.'
            )
            return

        # Distance: from current pose to the LLM-issued goal
        pose = self._current_pose
        gx = self._llm_ctx['goal_x_world']
        gy = self._llm_ctx['goal_y_world']
        self.distance = float(math.hypot(gx - pose.x, gy - pose.y))
        self.v_max    = self._llm_ctx['v_const']
        self.a_max    = self._llm_ctx['a']
        self._goal_x_world = gx
        self._goal_y_world = gy

        if self.distance < 1e-3 or self.v_max <= 0 or self.a_max <= 0:
            self.get_logger().warning(
                f'[trapezoid] degenerate constraints: distance={self.distance:.3f}, '
                f'v_const={self.v_max:.3f}, a={self.a_max:.3f}; ignoring.'
            )
            return

        self._compute_trapezoid_profile()
        self._t = 0.0
        self._start_pose = (pose.x, pose.y, pose.theta)

        # Publish the planned classical trajectory immediately so the plotter
        # can show the setpoint while motion is in progress (not just at end).
        heading_to_goal = math.atan2(gy - pose.y, gx - pose.x)
        planned = self._compute_classical_trajectory(
            start_xy=(pose.x, pose.y),
            heading=heading_to_goal,
            distance=self.distance,
            v_const=self.v_peak,
            a_lin=self.a_max,
        )
        self._publish_path(self._classical_path_pub, planned)

        # Re-engage the timer if it was cancelled after a previous completion
        self._ensure_timer_running()
        self._phase = self._ACCEL
        self.get_logger().info(
            f'[trapezoid] LLM constraints applied: '
            f'v_peak={self.v_peak:.4f} a={self.a_max:.4f} dist={self.distance:.3f} t_end={self.t3:.2f}s '
            f'profile={"triangular" if self._triangular else "trapezoidal"}'
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
        self.goal_x: float | None = None
        self.goal_y: float | None = None
        self.goal_tolerance = float(self.get_parameter('goal_tolerance').value)
        self._plan_once = bool(self.get_parameter('plan_once').value)
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

        # ── CFM 3D model (loaded once at startup) ────────────────────────
        self.get_logger().info('[PID] Loading CFM 3D model …')
        self.diffusion = self.model_init()
        self.get_logger().info('[PID] CFM model ready.')

        # ── trajectory state (50 waypoints of x, y, theta) ───────────────
        self._traj_waypoints  = None   # np.ndarray (50, 3)
        self._traj_idx        = 0
        self._traj_step_timer = 0.0    # time accumulator for 25 Hz waypoint stepping
        self._replan_timer    = 0.0    # time accumulator for 1 Hz CFM replanning
        # Snapshot of constraints used for the current trajectory (for logging)
        self._active_v_const     = V_CONST_FALLBACK
        self._active_a           = A_FALLBACK
        self._active_start       = (0.0, 0.0, 0.0)
        self._active_resample    = 0
        self._active_context_13dim = np.zeros(13, dtype=np.float32)
        # Safety: cap resamples per goal so we don't spin forever when CFM
        # produces unreliable output for a particular goal direction. Reset
        # on every new goal in _apply_pid_context / _goal_cb.
        self._resample_count = 0

        # Manual goal topic (backward compat / quick testing)
        self.create_subscription(
            Point, 'goal_position', self._goal_cb, 10
        )

        self.get_logger().info(
            f'\n  [pid]\n'
            f'  goal            = waiting for /traj_context (LLM) or /goal_position\n'
            f'  tolerance       = {self.goal_tolerance:.3f} m\n'
            f'  plan_once       = {self._plan_once}  ({"no replanning" if self._plan_once else "1 Hz replanning"})\n'
            f'  linear  PID     : Kp={self.pid_linear.kp}  '
            f'Ki={self.pid_linear.ki}  Kd={self.pid_linear.kd}\n'
            f'  angular PID     : Kp={self.pid_angular.kp}  '
            f'Ki={self.pid_angular.ki}  Kd={self.pid_angular.kd}'
        )

    def _apply_pid_context(self):
        """Activate a new goal from /traj_context in PID/CFM mode."""
        if self._llm_ctx is None or self._current_pose is None:
            return
        self.goal_x = self._llm_ctx['goal_x_world']
        self.goal_y = self._llm_ctx['goal_y_world']
        self.pid_linear.reset()
        self.pid_angular.reset()
        self._resample_count = 0
        self._replan_timer   = 0.0
        # Re-engage timer if completed previously
        self._ensure_timer_running()
        # Resample BEFORE flipping to ACTIVE so the timer sees a valid trajectory
        self._resample_trajectory(self._current_pose, label='LLM-context')
        # Publish the analytical-classical setpoint ONCE per goal as a comparison
        # baseline (CFM is the active controller; this is plot-only).
        self._publish_classical_baseline()
        self._phase = self._ACTIVE
        self.get_logger().info(
            f'[PID] LLM goal applied: ({self.goal_x:.3f}, {self.goal_y:.3f}); CFM resampled'
        )

    def _publish_classical_baseline(self):
        """Compute and publish the analytical classical (start → goal) trajectory.
        Called once per goal application so the plot's classical baseline
        doesn't shift around as the controller resamples CFM."""
        if self.goal_x is None or self.goal_y is None:
            return
        # Use the pose snapshotted at context-receipt time, not self._current_pose,
        # which may have drifted during CFM model inference (~200-500 ms).
        if self._llm_ctx and 'context_wx' in self._llm_ctx:
            sx, sy = self._llm_ctx['context_wx'], self._llm_ctx['context_wy']
        elif self._current_pose is not None:
            sx, sy = self._current_pose.x, self._current_pose.y
        else:
            return
        gx, gy = float(self.goal_x), float(self.goal_y)
        dist = math.hypot(gx - sx, gy - sy)
        if dist < 1e-6:
            return
        v = self._llm_ctx['v_const'] if self._llm_ctx else V_CONST_FALLBACK
        a = self._llm_ctx['a']       if self._llm_ctx else A_FALLBACK
        traj = self._compute_classical_trajectory(
            start_xy=(sx, sy),
            heading=math.atan2(gy - sy, gx - sx),
            distance=dist,
            v_const=v,
            a_lin=a,
        )
        self._publish_path(self._classical_path_pub, traj)

    def _goal_cb(self, msg: Point):
        # Manual goal override (no LLM constraints — uses cached or fallback)
        if (self.goal_x is not None and self.goal_y is not None
                and abs(msg.x - self.goal_x) < 1e-3
                and abs(msg.y - self.goal_y) < 1e-3):
            return
        self.goal_x = msg.x
        self.goal_y = msg.y
        self.pid_linear.reset()
        self.pid_angular.reset()
        self._resample_count = 0
        self._replan_timer   = 0.0
        self._ensure_timer_running()
        if self._current_pose is None:
            self.get_logger().warning(
                f'[PID] Goal ({msg.x:.2f}, {msg.y:.2f}) received before first pose — '
                'CFM sample deferred until pose arrives.'
            )
        else:
            self._resample_trajectory(self._current_pose, label='manual-goal')
            self._publish_classical_baseline()
        self._phase = self._ACTIVE
        self.get_logger().info(
            f'[PID] Manual goal: ({self.goal_x:.3f}, {self.goal_y:.3f})'
        )

    # ── timer callback ────────────────────────────────────────────────────
    def _timer_cb(self):
        if self._phase == self._WAITING:
            self._wait += self.dt
            if self._wait >= self.start_delay:
                # On wakeup: if we have an LLM context, apply it; else use fallback
                if self.mode == 'trapezoid':
                    if self._llm_ctx is not None and self._current_pose is not None:
                        self._apply_trapezoid_context()
                    elif self._current_pose is not None:
                        self._start_pose = (
                            self._current_pose.x, self._current_pose.y, self._current_pose.theta
                        )
                        self._phase = self._ACCEL
                        self.get_logger().info('[trapezoid] Using launch-param fallback.')
                    else:
                        # No pose yet — keep waiting briefly
                        self._wait = self.start_delay - 0.1
                        return
                else:
                    # PID/CFM: only activate if we already have a goal (LLM context or
                    # manual /goal_position). Otherwise stay in WAITING — the next
                    # TrajContext / goal will activate via _apply_pid_context / _goal_cb.
                    if self._llm_ctx is not None and self._current_pose is not None:
                        self._apply_pid_context()
                    elif self.goal_x is not None and self.goal_y is not None:
                        self._phase = self._ACTIVE
                    else:
                        # Reset wait so we re-enter this branch on the next tick
                        self._wait = self.start_delay - self.dt
                        return
                self.get_logger().info(f'[{self.mode.upper()}] Control started.')
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
            self._save_trapezoid_log()

        self._t += self.dt

    # ── trajectory helper ─────────────────────────────────────────────────
    def _resample_trajectory(self, pose, label: str = 'sample'):
        # Use cached LLM constraints if present, else fallback
        if self._llm_ctx is not None:
            v_const = self._llm_ctx['v_const']
            a_lin   = self._llm_ctx['a']
        else:
            v_const = V_CONST_FALLBACK
            a_lin   = A_FALLBACK

        # Model uses local frame: current pose is always the origin (0, 0).
        # Goals and velocity derivatives are expressed relative to current pose.
        if pose is not None:
            q_theta      = pose.theta
            qdot_theta   = pose.angular_velocity
            pose_world_x = pose.x
            pose_world_y = pose.y
        else:
            q_theta = qdot_theta = 0.0
            pose_world_x = pose_world_y = 0.0

        # ── Body-frame normalization ──────────────────────────────────────
        # Model trained in body frame: robot always at (0,0) facing +x.
        # 1. Goal: translate then rotate by -theta into body frame
        cos_t = math.cos(q_theta)
        sin_t = math.sin(q_theta)
        dx = self.goal_x - pose_world_x
        dy = self.goal_y - pose_world_y
        q_x_m    = 0.0
        q_y_m    = 0.0
        goal_x_m =  dx * cos_t + dy * sin_t
        goal_y_m = -dx * sin_t + dy * cos_t
        s_goal_theta = math.atan2(goal_y_m, goal_x_m)

        # 2. Heading: always 0 in body frame (we absorbed theta into goal rotation)
        q_theta_model = 0.0

        # 3. Velocity: body frame (forward = v, lateral = 0)
        qdot_x = pose.linear_velocity if pose is not None else 0.0
        qdot_y = 0.0

        # Snapshot for logging
        self._active_v_const    = v_const
        self._active_a          = a_lin
        self._active_resample   = self._resample_count
        if pose is not None:
            self._active_start = (pose.x, pose.y, pose.theta)
        else:
            self._active_start = (0.0, 0.0, 0.0)

        # Context: [s_goal_x, s_goal_y, s_goal_theta,
        #          v_const, accel, omega_const, alpha_const,
        #          q_init_x, q_init_y, q_init_theta,
        #          qdot_init_x, qdot_init_y, qdot_init_theta]
        context_data = np.array(
            [[goal_x_m, goal_y_m, s_goal_theta,
              v_const, a_lin, OMEGA_CONST_DEFAULT, ALPHA_CONST_DEFAULT,
              q_x_m, q_y_m, q_theta_model,
              qdot_x, qdot_y, qdot_theta]],
            dtype=np.float32
        )
        self._active_context_13dim = context_data[0].copy()  # (13,) for logging
        context = torch.tensor(context_data, dtype=torch.float32)
        self.get_logger().info(
            f'[CFM-3D] Context: goal_body=({goal_x_m:.3f},{goal_y_m:.3f},{s_goal_theta:.3f}) '
            f'world_theta={q_theta:.3f} '
            f'v={v_const:.4f} a={a_lin:.4f} '
            f'qdot_body=({qdot_x:.3f},{qdot_y:.3f})'
        )
        _t0 = self.get_clock().now()
        samples, _ = self.sample_trajectories(self.diffusion, context)
        _dt_ms = (self.get_clock().now() - _t0).nanoseconds / 1e6
        # Publish gen latency so the plotter can record it to CSV (no log parsing needed)
        self._cfm_gen_ms_pub.publish(Float32(data=float(_dt_ms)))

        waypoints = samples[0]          # (50, 3): x, y, theta (meter frame)
        self.get_logger().info(
            f'[CFM-3D] Trajectory {label} in {_dt_ms:.1f} ms '
            f'(v_const={v_const:.4f}, a={a_lin:.4f}) | '
            f'raw_wpts[0]=({waypoints[0,0]:.3f},{waypoints[0,1]:.3f}) '
            f'raw_wpts[-1]=({waypoints[-1,0]:.3f},{waypoints[-1,1]:.3f}) '
            f'total_span_xy=({waypoints[-1,0]-waypoints[0,0]:.3f},{waypoints[-1,1]-waypoints[0,1]:.3f})'
        )

        # Anchor: model output starts near (0,0) body frame but not exactly.
        # Shift so waypoints[0] = (0, 0), preserving trajectory shape.
        waypoints[:, 0] -= waypoints[0, 0]
        waypoints[:, 1] -= waypoints[0, 1]

        # 4. Convert body frame → world:
        #    rotate by +theta, then translate by current world position.
        waypoints_world = waypoints.copy()
        bx = waypoints[:, 0].copy()
        by = waypoints[:, 1].copy()
        waypoints_world[:, 0] = bx * cos_t - by * sin_t + pose_world_x
        waypoints_world[:, 1] = bx * sin_t + by * cos_t + pose_world_y
        waypoints_world[:, 2] = waypoints[:, 2] + q_theta  # rotate heading too
        self._traj_waypoints  = waypoints_world
        self._traj_idx        = 0
        self._traj_step_timer = 0.0

        # Diagnostic logging only — does NOT change behavior. The project's
        # purpose is to show whether CFM produces correct setpoints; the
        # turtle blindly follows whatever CFM emits, so failures are visible
        # on the plot (turtle drifts away, doesn't reach goal, etc.).
        if pose is not None and self.goal_x is not None and self.goal_y is not None:
            end_x = float(waypoints_world[-1, 0])
            end_y = float(waypoints_world[-1, 1])
            start_to_goal = math.hypot(self.goal_x - pose.x,  self.goal_y - pose.y)
            end_to_goal   = math.hypot(self.goal_x - end_x,   self.goal_y - end_y)
            progress_made = start_to_goal - end_to_goal
            if start_to_goal > 0.10 and end_to_goal > start_to_goal * 1.10:
                self.get_logger().warning(
                    f'[CFM] Trajectory DIVERGING from goal '
                    f'(start→goal={start_to_goal:.2f} m, end→goal={end_to_goal:.2f} m, '
                    f'progress={progress_made:+.2f} m). Turtle will follow it anyway — '
                    f'plot shows the failure.'
                )
            elif start_to_goal > 0.5 and progress_made < 0.30 * start_to_goal:
                self.get_logger().warning(
                    f'[CFM] Low goalward progress '
                    f'(start→goal={start_to_goal:.2f} m, end→goal={end_to_goal:.2f} m, '
                    f'progress={progress_made:+.2f} m). Turtle will follow it.'
                )

        # Publish CFM setpoint so the plotter can overlay it
        self._publish_path(self._cfm_path_pub, waypoints_world)

        # Also save the per-trajectory comparison log (CFM + analytical classical)
        # _save_pid_log additionally publishes the analytical-classical setpoint.
        self._save_pid_log(cfm_waypoints_world=waypoints_world)

    # ── analytical classical trapezoid trajectory ────────────────────────
    def _compute_classical_trajectory(
        self,
        start_xy: tuple[float, float],
        heading: float,
        distance: float,
        v_const: float,
        a_lin: float,
        dt: float = CFM_DT,
    ) -> np.ndarray:
        """Integrate (x, y, theta) of a straight-line trapezoidal motion.

        Drives `distance` meters along `heading` from `start_xy`, with peak
        velocity `v_const` and acceleration `a_lin`. Returns (N, 3) at `dt`.
        Used purely for logging / offline comparison."""
        x0, y0 = start_xy
        if distance < 1e-6 or v_const <= 0 or a_lin <= 0:
            return np.array([[x0, y0, heading]], dtype=np.float32)

        t_accel = v_const / a_lin
        d_accel = 0.5 * a_lin * t_accel ** 2
        if 2.0 * d_accel >= distance:
            v_peak = math.sqrt(a_lin * distance)
            t_ramp = v_peak / a_lin
            t1 = t_ramp; t2 = t_ramp; t3 = 2.0 * t_ramp
        else:
            v_peak = v_const
            d_const = distance - 2.0 * d_accel
            t1 = t_accel
            t2 = t_accel + d_const / v_const
            t3 = t2 + t_accel

        n_steps = int(math.ceil(t3 / dt)) + 1
        traj = np.zeros((n_steps, 3), dtype=np.float32)
        s = 0.0
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        for i in range(n_steps):
            t = i * dt
            if t <= t1:
                v = a_lin * t
            elif t <= t2:
                v = v_peak
            elif t <= t3:
                v = v_peak - a_lin * (t - t2)
            else:
                v = 0.0
            if i > 0:
                s += v * dt
            traj[i, 0] = x0 + s * cos_h
            traj[i, 1] = y0 + s * sin_h
            traj[i, 2] = heading
        return traj

    # ── comparison logging ────────────────────────────────────────────────
    def _save_pid_log(self, cfm_waypoints_world: np.ndarray):
        """In PID/CFM mode, save BOTH the CFM trajectory and the analytical
        classical trapezoid trajectory computed from the same constraints
        (heading: from start toward goal; distance: ||goal - start||)."""
        try:
            # Classical is always start→goal in one shot, computed once from
            # the pose at context-receipt time — never from the moving resample pos.
            if self._llm_ctx and 'context_wx' in self._llm_ctx:
                sx, sy = self._llm_ctx['context_wx'], self._llm_ctx['context_wy']
            else:
                sx, sy, _ = self._active_start
            gx, gy = float(self.goal_x), float(self.goal_y)
            heading_to_goal = math.atan2(gy - sy, gx - sx)
            dist_to_goal = math.hypot(gx - sx, gy - sy)
            classical = self._compute_classical_trajectory(
                start_xy=(sx, sy),
                heading=heading_to_goal,
                distance=dist_to_goal,
                v_const=self._active_v_const,
                a_lin=self._active_a,
            )
            # NOTE: classical setpoint is published once per goal in
            # _apply_pid_context / _goal_cb (not per resample) so the plot
            # keeps a stable baseline. We still save it to the npz log.
            ts = time.strftime('%Y%m%d-%H%M%S') + f'-{int((time.time() % 1) * 1000):03d}'
            path = os.path.join(LOG_DIR, f'{ts}_pid.npz')
            ctx_13 = getattr(self, '_active_context_13dim', np.zeros(13, dtype=np.float32))
            np.savez(
                path,
                # trajectories
                cfm_waypoints=cfm_waypoints_world.astype(np.float32),   # (50, 3) world frame
                classical_waypoints=classical,                           # (N, 3) world frame
                # timing
                cfm_dt=np.float32(CFM_DT),
                classical_dt=np.float32(CFM_DT),
                # kinematics
                v_const=np.float32(self._active_v_const),
                a=np.float32(self._active_a),
                # goal / pose
                goal_world=np.array([self.goal_x, self.goal_y], dtype=np.float32),
                start_pose=np.array(self._active_start, dtype=np.float32),
                # model input — 13-dim context vector fed to CFM
                # [goal_x_local, goal_y_local, s_goal_theta,
                #  v_const, a, omega_const, alpha_const,
                #  q_x=0, q_y=0, q_theta, qdot_x, qdot_y, qdot_theta]
                context_13dim=ctx_13,
                # resample bookkeeping
                resample_idx=np.int32(self._active_resample),
            )
            self.get_logger().info(f'[log] Saved comparison → {path}')
        except Exception as e:
            self.get_logger().warning(f'[log] save failed: {e}')

    def _save_trapezoid_log(self):
        """In trapezoid mode, save the analytical classical baseline toward the
        LLM-issued goal so this log is directly comparable with pid-mode logs.
        (The actually-executed turtle motion may differ since trapezoid mode is
        an open-loop 1D primitive that drives along its current heading.)"""
        try:
            if self._start_pose is None:
                return
            sx, sy, _ = self._start_pose
            gx, gy = self._goal_x_world, self._goal_y_world
            heading_to_goal = math.atan2(gy - sy, gx - sx)
            dist_to_goal = math.hypot(gx - sx, gy - sy)
            classical = self._compute_classical_trajectory(
                start_xy=(sx, sy),
                heading=heading_to_goal,
                distance=dist_to_goal,
                v_const=self.v_peak,
                a_lin=self.a_max,
            )
            # Publish classical setpoint for plotter overlay
            self._publish_path(self._classical_path_pub, classical)
            ts = time.strftime('%Y%m%d-%H%M%S') + f'-{int((time.time() % 1) * 1000):03d}'
            path = os.path.join(LOG_DIR, f'{ts}_trapezoid.npz')
            np.savez(
                path,
                classical_waypoints=classical,
                classical_dt=np.float32(CFM_DT),
                v_const=np.float32(self.v_peak),
                a=np.float32(self.a_max),
                goal_world=np.array([self._goal_x_world, self._goal_y_world], dtype=np.float32),
                start_pose=np.array(self._start_pose, dtype=np.float32),
                t_end=np.float32(self.t3),
            )
            self.get_logger().info(f'[log] Saved trapezoid trajectory → {path}')
        except Exception as e:
            self.get_logger().warning(f'[log] save failed: {e}')

    # ── model ─────────────────────────────────────────────────────────────
    def model_init(self):
        import sys
        scripts_dir = os.path.dirname(os.path.realpath(__file__))
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        from diffuser.models.temporal_film import ConditionalUnet1D
        from diffuser.models.cfm import CFM

        observation_dim = 3   # x, y, theta
        action_dim      = 0
        context_dim     = 13  # see schema in _resample_trajectory()

        model = ConditionalUnet1D(
            horizon=HORIZON,
            transition_dim=observation_dim + action_dim,
            lstm_in_dim=None,
            lstm_out_dim=None,
            global_cond_dim=context_dim,
            cond_dim=observation_dim,
            dim_mults=(1, 4, 8),
        ).to(DEVICE)

        diffusion = CFM(
            model=model,
            horizon=HORIZON,
            observation_dim=observation_dim,
            action_dim=action_dim,
            n_timesteps=N_SAMPLING_STEPS,
            loss_type='l2',
            predict_epsilon=False,
        ).to(DEVICE)

        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        raw = checkpoint['model']
        prefix = 'node.vf.vf.'
        remapped = {('model.' + k[len(prefix):] if k.startswith(prefix) else k): v for k, v in raw.items()}
        diffusion.load_state_dict(remapped)
        return diffusion

    @torch.no_grad()
    def sample_trajectories(self, diffusion, contexts, n_samples_per=1):
        diffusion.eval()
        ctx        = contexts.repeat_interleave(n_samples_per, dim=0).to(DEVICE)
        batch_size = ctx.shape[0]
        global_cond = {'hideouts': ctx}
        cond        = [(np.array([]), np.array([]))] * batch_size
        samples     = diffusion.conditional_sample(global_cond, cond)  # (B, HORIZON, 3)
        samples     = samples.cpu().numpy()[:, :ORIGINAL_LEN, :]        # (B, 50, 3)
        return samples, ctx.cpu().numpy()

    # ── PID step ──────────────────────────────────────────────────────────
    # CFM-only tracking. No classical / direct-PID fallback — the project's
    # whole point is observing how flow matching performs end-to-end. To keep
    # the controller from spinning forever when CFM cannot reach a particular
    # goal, we cap the number of resamples per goal.
    _MAX_RESAMPLES_PER_GOAL = 30

    def _step_pid(self):
        if self._current_pose is None or self.goal_x is None or self.goal_y is None:
            return
        if self._traj_waypoints is None:
            # Pose has just arrived — trigger the resample that was deferred in _goal_cb
            self.get_logger().info('[PID] Deferred CFM resample triggered (pose now available).')
            self._resample_trajectory(self._current_pose, label='deferred')
            self._publish_classical_baseline()
            if self._traj_waypoints is None:
                return  # resample failed (e.g. model not loaded yet)

        pose = self._current_pose

        # --- goal reached ---
        dist_to_goal = math.hypot(self.goal_x - pose.x, self.goal_y - pose.y)
        if dist_to_goal < self.goal_tolerance:
            self._publish(0.0, 0.0)
            self._phase = self._DONE
            self.get_logger().info(
                f'[PID] Goal reached!  pos=({pose.x:.3f}, {pose.y:.3f})  '
                f'error={dist_to_goal:.4f} m  — waiting for next /traj_context or /goal_position'
            )
            return

        # --- 25 Hz waypoint stepping ---
        # Advance one waypoint every CFM_DT=0.04 s (25 Hz) so playback
        # matches the temporal structure the model was trained on.
        self._traj_step_timer += self.dt
        if self._traj_step_timer >= CFM_DT:
            self._traj_step_timer = 0.0
            self._traj_idx = min(self._traj_idx + 1, ORIGINAL_LEN - 1)
            self.get_logger().debug(f'[PID] Waypoint {self._traj_idx}/{ORIGINAL_LEN}')

        # --- 1 Hz CFM replanning (disabled when plan_once=True) ---
        # Every 1 second regenerate a fresh trajectory from current pose to goal.
        # This keeps the plan up-to-date as the turtle moves.
        if not self._plan_once:
            self._replan_timer += self.dt
            if self._replan_timer >= 1.0:
                self._replan_timer = 0.0
                self._resample_count += 1
                if self._resample_count > self._MAX_RESAMPLES_PER_GOAL:
                    self._publish(0.0, 0.0)
                    self._phase = self._DONE
                    self.get_logger().warning(
                        f'[CFM] Stopped: max resamples ({self._MAX_RESAMPLES_PER_GOAL}) '
                        f'reached without converging on goal. Final error: {dist_to_goal:.2f} m.'
                    )
                    return
                self._resample_trajectory(pose, label=f'replan#{self._resample_count}')

        target_x     = self._traj_waypoints[self._traj_idx, 0]
        target_y     = self._traj_waypoints[self._traj_idx, 1]
        target_theta = self._traj_waypoints[self._traj_idx, 2]   # CFM-predicted heading

        # Linear: PID on distance to the current waypoint position
        dx   = target_x - pose.x
        dy   = target_y - pose.y
        dist = math.hypot(dx, dy)
        v    = self.pid_linear.compute(dist, self.dt)

        # Angular: track CFM's predicted theta directly (uses model's full 3D output)
        angle_error = target_theta - pose.theta
        angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi
        w = self.pid_angular.compute(angle_error, self.dt)

        # Scale down linear speed proportional to heading alignment
        heading_factor = max(0.0, math.cos(angle_error))
        v *= heading_factor

        self._publish(v, w)

    # ── helpers ───────────────────────────────────────────────────────────
    def _publish(self, linear: float, angular: float = 0.0):
        msg = Twist()
        msg.linear.x  = float(linear)
        msg.angular.z = float(angular)
        self.cmd_vel_pub.publish(msg)

    def _publish_path(self, publisher, waypoints: np.ndarray):
        """Publish a (N, 3) ndarray of (x, y, theta) world-coord waypoints
        as a geometry_msgs/PoseArray on the given publisher. Used so the
        turtle_plotter can overlay setpoint trajectories on the XY plot."""
        if waypoints is None or len(waypoints) == 0:
            return
        msg = PoseArray()
        msg.header.frame_id = 'world'
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y, th in waypoints:
            p = GeomPose()
            p.position.x = float(x)
            p.position.y = float(y)
            p.position.z = 0.0
            # Encode theta as a yaw-only quaternion (z, w only)
            p.orientation.z = math.sin(float(th) / 2.0)
            p.orientation.w = math.cos(float(th) / 2.0)
            msg.poses.append(p)
        publisher.publish(msg)

    def _ensure_timer_running(self):
        """Re-arm the periodic timer if it was cancelled at trajectory completion.
        Destroys the old timer first to avoid leaking it in the node's timer list."""
        if self._timer is None:
            self._timer = self.create_timer(self.dt, self._timer_cb)
            return
        if self._timer.is_canceled():
            self.destroy_timer(self._timer)
            self._timer = self.create_timer(self.dt, self._timer_cb)


# ── entry point ──────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = TurtleController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
