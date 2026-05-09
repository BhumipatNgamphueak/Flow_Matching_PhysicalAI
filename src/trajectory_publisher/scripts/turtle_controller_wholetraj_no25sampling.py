# #!/usr/bin/python3

# """
# turtle_controller.py
# ---------------------
# Unified turtle controller with two selectable modes:

#   mode = 'trapezoid'  →  Publishes a trapezoidal (or triangular) linear
#                           velocity profile for straight-line motion.

#   mode = 'pid'        →  PID position controller that drives the turtle
#                           to a goal position received via the topic
#                           /goal_position (geometry_msgs/Point).
#                           Publish a new point at any time to update the goal.

#   ros2 launch trajectory_publisher turtlesim_trapezoid.launch.py mode:=pid
#   ros2 launch trajectory_publisher turtlesim_trapezoid.launch.py mode:=trapezoid distance:=6.0

#   # Send a goal manually:
#   ros2 topic pub /goal_position geometry_msgs/Point "{x: 10.0, y: 7.5, z: 0.0}"
# """

# import math
# import numpy as np
# import torch
# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import Twist, Point
# from turtlesim.msg import Pose

# CHECKPOINT_PATH = "/home/bubble/PhysicalAI_flowmatching/logs/pose_trajectory_athit_notrajpart/cfm/H64_T100/20260416-1405/state_192000.pt"
# TURTLESIM_ORIGIN_X = 5.544   # TurtleSim default spawn x
# TURTLESIM_ORIGIN_Y = 5.544   # TurtleSim default spawn y
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# HORIZON = 64        # padded horizon (must match training)
# ORIGINAL_LEN = 50   # original trajectory length
# N_SAMPLING_STEPS = 10    # Euler sampling steps for CFM (flow matching needs far fewer than diffusion)
# CFM_DT           = 0.04  # seconds per waypoint (25 Hz playback)

# # ── PID helper ──────────────────────────────────────────────────────────────
# class PIDController:
#     """Simple discrete PID with anti-windup clamp."""

#     def __init__(self, kp: float, ki: float, kd: float,
#                  output_min: float = -float('inf'),
#                  output_max: float =  float('inf')):
#         self.kp = kp
#         self.ki = ki
#         self.kd = kd
#         self.output_min = output_min
#         self.output_max = output_max
#         self._integral   = 0.0
#         self._prev_error = 0.0

#     def compute(self, error: float, dt: float) -> float:
#         if dt <= 0:
#             return 0.0
#         self._integral  += error * dt
#         derivative       = (error - self._prev_error) / dt
#         self._prev_error = error
#         raw = self.kp * error + self.ki * self._integral + self.kd * derivative
#         return float(max(self.output_min, min(self.output_max, raw)))

#     def reset(self):
#         self._integral   = 0.0
#         self._prev_error = 0.0


# # ── Main node ────────────────────────────────────────────────────────────────
# class TurtleController(Node):

#     # State machine labels (shared by both modes)
#     _WAITING  = 'WAITING'
#     _ACCEL    = 'ACCELERATING'   # trapezoid only
#     _CONSTANT = 'CONSTANT'       # trapezoid only
#     _DECEL    = 'DECELERATING'   # trapezoid only
#     _ACTIVE   = 'ACTIVE'         # pid only
#     _DONE     = 'DONE'

#     def __init__(self):
#         super().__init__('turtle_controller')

#         # ── common parameters ────────────────────────────────────────────
#         self.declare_parameter('mode',        'trapezoid')   # 'trapezoid' | 'pid'
#         self.declare_parameter('turtle_name', 'turtle1')
#         self.declare_parameter('dt',           0.01)         # timer period [s]
#         self.declare_parameter('start_delay',  2.0)          # wait before moving [s]

#         # ── trapezoid parameters ─────────────────────────────────────────
#         self.declare_parameter('v_max',    2.0)   # m/s
#         # self.declare_parameter('v_max',   0.156)   # m/s
#         self.declare_parameter('a_max',    0.5)   # m/s²
#         # self.declare_parameter('a_max',   0.028)   # m/s²
#         self.declare_parameter('distance', 5.0)   # m

#         # ── PID parameters ───────────────────────────────────────────────
#         self.declare_parameter('goal_tolerance',  0.05)   # m
#         # self.declare_parameter('v_max_pid',       2.0)    # m/s  linear clamp
#         self.declare_parameter('v_max_pid',       0.156)    # m/s  linear clamp
#         self.declare_parameter('w_max_pid',       3.0)    # rad/s angular clamp
#         self.declare_parameter('kp_linear',       1.5)
#         self.declare_parameter('ki_linear',       0.0)
#         self.declare_parameter('kd_linear',       0.05)
#         self.declare_parameter('kp_angular',      5.0)
#         self.declare_parameter('ki_angular',      0.0)
#         self.declare_parameter('kd_angular',      0.1)
#         self.declare_parameter('angle_tolerance',    0.05)   # rad — rotation done threshold
#         self.declare_parameter('waypoint_tolerance', 0.15)  # m   — advance to next waypoint

#         # ── read common params ────────────────────────────────────────────
#         self.mode        = self.get_parameter('mode').value
#         self.turtle_name = self.get_parameter('turtle_name').value
#         self.dt          = float(self.get_parameter('dt').value)
#         self.start_delay = float(self.get_parameter('start_delay').value)

#         # ── publisher ────────────────────────────────────────────────────
#         self.cmd_vel_pub = self.create_publisher(
#             Twist, f'{self.turtle_name}/cmd_vel', 10
#         )

#         # ── mode-specific init ────────────────────────────────────────────
#         if self.mode == 'trapezoid':
#             self._init_trapezoid()
#         elif self.mode == 'pid':
#             self._init_pid()
#         else:
#             self.get_logger().fatal(
#                 f"Unknown mode '{self.mode}'. Use 'trapezoid' or 'pid'."
#             )
#             return

#         # ── runtime state ─────────────────────────────────────────────────
#         self._wait  = 0.0
#         self._phase = self._WAITING
#         self._timer = self.create_timer(self.dt, self._timer_cb)

#         self.get_logger().info(
#             f'\n┌─ TurtleController ready ─────────────────────\n'
#             f'│  mode        : {self.mode}\n'
#             f'│  turtle      : {self.turtle_name}\n'
#             f'│  start delay : {self.start_delay} s\n'
#             f'└──────────────────────────────────────────────'
#         )

#     # ── trapezoid init ────────────────────────────────────────────────────
#     def _init_trapezoid(self):
#         self.v_max    = float(self.get_parameter('v_max').value)
#         self.a_max    = float(self.get_parameter('a_max').value)
#         self.distance = float(self.get_parameter('distance').value)
#         self._t = 0.0
#         self._compute_trapezoid_profile()

#         self.get_logger().info(
#             f'\n  [trapezoid]\n'
#             f'  v_max    = {self.v_peak:.3f} m/s\n'
#             f'  a_max    = {self.a_max:.3f} m/s²\n'
#             f'  distance = {self.distance:.3f} m\n'
#             f'  t_end    = {self.t3:.3f} s\n'
#             f'  profile  = {"triangular" if self._triangular else "trapezoidal"}'
#         )

#     def _compute_trapezoid_profile(self):
#         t_accel = self.v_max / self.a_max
#         d_accel = 0.5 * self.a_max * t_accel ** 2
#         if 2.0 * d_accel >= self.distance:
#             self._triangular = True
#             self.v_peak = math.sqrt(self.a_max * self.distance)
#             t_ramp = self.v_peak / self.a_max
#             self.t1 = t_ramp
#             self.t2 = t_ramp
#             self.t3 = 2.0 * t_ramp
#         else:
#             self._triangular = False
#             self.v_peak = self.v_max
#             d_const = self.distance - 2.0 * d_accel
#             self.t1 = t_accel
#             self.t2 = t_accel + d_const / self.v_max
#             self.t3 = self.t2 + t_accel

#     def _velocity_at(self, t: float) -> float:
#         if t <= self.t1:
#             return self.a_max * t
#         elif t <= self.t2:
#             return self.v_peak
#         elif t <= self.t3:
#             return self.v_peak - self.a_max * (t - self.t2)
#         return 0.0

#     # ── PID init ──────────────────────────────────────────────────────────
#     def _init_pid(self):
#         self.goal_x: float | None = None   # set by /goal_position topic
#         self.goal_y: float | None = None
#         self.goal_tolerance = float(self.get_parameter('goal_tolerance').value)
#         v_lim = float(self.get_parameter('v_max_pid').value)
#         w_lim = float(self.get_parameter('w_max_pid').value)

#         self.pid_linear = PIDController(
#             kp=float(self.get_parameter('kp_linear').value),
#             ki=float(self.get_parameter('ki_linear').value),
#             kd=float(self.get_parameter('kd_linear').value),
#             output_min=-v_lim, output_max=v_lim,
#         )
#         self.pid_angular = PIDController(
#             kp=float(self.get_parameter('kp_angular').value),
#             ki=float(self.get_parameter('ki_angular').value),
#             kd=float(self.get_parameter('kd_angular').value),
#             output_min=-w_lim, output_max=w_lim,
#         )

#         self.angle_tolerance    = float(self.get_parameter('angle_tolerance').value)
#         self.waypoint_tolerance = float(self.get_parameter('waypoint_tolerance').value)
#         self._rotating = False   # True while doing rotate-first phase

#         # ── CFM model (loaded once at startup) ───────────────────────────
#         self.get_logger().info('[PID] Loading CFM model …')
#         self.diffusion = self.model_init()
#         self.get_logger().info('[PID] CFM model ready.')

#         # ── trajectory state ─────────────────────────────────────────────
#         self._traj_waypoints  = None   # np.ndarray (50, 2) after sampling
#         self._traj_idx        = 0      # index of current waypoint
#         self._traj_step_timer = 0.0    # time spent at current waypoint

#         self._current_pose: Pose | None = None
#         self.create_subscription(
#             Pose, f'{self.turtle_name}/pose', self._pose_cb, 10
#         )
#         self.create_subscription(
#             Point, 'goal_position', self._goal_cb, 10
#         )

#         self.get_logger().info(
#             f'\n  [pid]\n'
#             f'  goal            = waiting for /goal_position topic\n'
#             f'  tolerance       = {self.goal_tolerance:.3f} m\n'
#             f'  linear  PID     : Kp={self.pid_linear.kp}  '
#             f'Ki={self.pid_linear.ki}  Kd={self.pid_linear.kd}\n'
#             f'  angular PID     : Kp={self.pid_angular.kp}  '
#             f'Ki={self.pid_angular.ki}  Kd={self.pid_angular.kd}'
#         )

#     def _pose_cb(self, msg: Pose):
#         self._current_pose = msg

#     def _goal_cb(self, msg: Point):
#         # Ignore repeated publishes of the same goal (ros2 topic pub streams at 1 Hz by default)
#         if (self.goal_x is not None and self.goal_y is not None
#                 and abs(msg.x - self.goal_x) < 1e-3
#                 and abs(msg.y - self.goal_y) < 1e-3):
#             return

#         self.goal_x = msg.x
#         self.goal_y = msg.y
#         self.pid_linear.reset()
#         self.pid_angular.reset()
#         self._rotating = True   # rotate first before translating
#         self._phase = self._ACTIVE

#         # ── sample initial CFM trajectory ────────────────────────────────
#         self._resample_trajectory(self._current_pose, label='initial')

#         self.get_logger().info(
#             f'[PID] New goal: ({self.goal_x:.3f}, {self.goal_y:.3f}) m — '
#             f'CFM trajectory sampled ({ORIGINAL_LEN} waypoints) — rotating first'
#         )

#     # ── timer callback ────────────────────────────────────────────────────
#     def _timer_cb(self):
#         # --- wait before starting ---
#         if self._phase == self._WAITING:
#             self._wait += self.dt
#             if self._wait >= self.start_delay:
#                 self._phase = self._ACCEL if self.mode == 'trapezoid' else self._ACTIVE
#                 self.get_logger().info(
#                     f'[{self.mode.upper()}] Control started.'
#                 )
#             return

#         if self._phase == self._DONE:
#             return

#         if self.mode == 'trapezoid':
#             self._step_trapezoid()
#         else:
#             self._step_pid()

#     # ── trapezoid step ────────────────────────────────────────────────────
#     def _step_trapezoid(self):
#         v = self._velocity_at(self._t)
#         self._publish(v, 0.0)

#         if self._phase == self._ACCEL and self._t > self.t1:
#             self._phase = self._CONSTANT if not self._triangular else self._DECEL
#             self.get_logger().info(
#                 f'Phase → {"CONSTANT" if not self._triangular else "DECELERATING"}'
#             )
#         elif self._phase == self._CONSTANT and self._t > self.t2:
#             self._phase = self._DECEL
#             self.get_logger().info('Phase → DECELERATING')
#         elif self._t > self.t3:
#             self._publish(0.0, 0.0)
#             self._phase = self._DONE
#             self._timer.cancel()
#             self.get_logger().info('Trapezoid trajectory complete.')

#         self._t += self.dt

#     # ── PID step ──────────────────────────────────────────────────────────
#     def model_init(self):
#         import sys, os
#         scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
#         if scripts_dir not in sys.path:
#             sys.path.insert(0, scripts_dir)

#         from diffuser.models.temporal_film import ConditionalUnet1D
#         from diffuser.models.cfm import CFM

#         observation_dim = 2
#         action_dim = 0
#         context_dim = 8

#         # Build model (must match training config)
#         model = ConditionalUnet1D(
#             horizon=HORIZON,
#             transition_dim=observation_dim + action_dim,
#             lstm_in_dim=None,
#             lstm_out_dim=None,
#             global_cond_dim=context_dim,
#             cond_dim=observation_dim,
#             dim_mults=(1, 4, 8),
#         ).to(DEVICE)

#         # Build diffusion wrapper
#         diffusion = CFM(
#             model=model,
#             horizon=HORIZON,
#             observation_dim=observation_dim,
#             action_dim=action_dim,
#             n_timesteps=N_SAMPLING_STEPS,
#             loss_type='l2',
#             predict_epsilon=False,
#         ).to(DEVICE)

#         # Load checkpoint — remap keys if saved with NeuralODE wrapper (node.vf.vf.* → model.*)
#         checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
#         raw_sd = checkpoint['model']
#         if any(k.startswith('node.vf.vf.') for k in raw_sd):
#             raw_sd = {k.replace('node.vf.vf.', 'model.', 1): v
#                       for k, v in raw_sd.items()
#                       if k.startswith('node.vf.vf.')}
#         diffusion.load_state_dict(raw_sd)
        
#         return diffusion
    
#     @torch.no_grad()
#     def sample_trajectories(self, diffusion, contexts, n_samples_per=1):
#         """
#         Generate trajectories for given context vectors.
        
#         Args:
#             diffusion: trained CFM model
#             contexts: (B, 9) context tensor
#             n_samples_per: how many samples to generate per context
#         Returns:
#             samples: (B * n_samples_per, ORIGINAL_LEN, 2) numpy array
#             contexts_repeated: (B * n_samples_per, 9) numpy array
#         """
        
#         diffusion.eval()
        
#         # repeat contexts for multiple samples
#         ctx = contexts.repeat_interleave(n_samples_per, dim=0).to(DEVICE)
#         batch_size = ctx.shape[0]
        
#         global_cond = {'hideouts': ctx}
#         cond = [(np.array([]), np.array([]))] * batch_size
        
#         samples = diffusion.conditional_sample(global_cond, cond)  # (B*n, HORIZON, 2)
#         samples = samples.cpu().numpy()
        
#         # trim padding back to original 50 steps
#         samples = samples[:, :ORIGINAL_LEN, :]
#         contexts_out = ctx.cpu().numpy()
        
#         return samples, contexts_out
    
    
#     def _resample_trajectory(self, pose, label: str = 'sample'):
#         if pose is not None:
#             qdot_x = pose.linear_velocity * math.cos(pose.theta)
#             qdot_y = pose.linear_velocity * math.sin(pose.theta)
#             q_x_m  = pose.x - TURTLESIM_ORIGIN_X
#             q_y_m  = pose.y - TURTLESIM_ORIGIN_Y
#         else:
#             qdot_x = qdot_y = 0.0
#             q_x_m  = q_y_m  = 0.0

#         goal_x_m = self.goal_x - TURTLESIM_ORIGIN_X
#         goal_y_m = self.goal_y - TURTLESIM_ORIGIN_Y

#         context_data = np.array(
#             [[goal_x_m, goal_y_m, qdot_x, qdot_y, 0.156, 0.028, q_x_m, q_y_m]],
#             dtype=np.float32
#         )
#         context = torch.tensor(context_data, dtype=torch.float32)
#         _t0 = self.get_clock().now()
#         samples, _ = self.sample_trajectories(self.diffusion, context)
#         _dt_ms = (self.get_clock().now() - _t0).nanoseconds / 1e6
#         self.get_logger().info(f'[CFM] Trajectory {label} in {_dt_ms:.1f} ms')

#         waypoints = samples[0]   # (ORIGINAL_LEN, 2)
#         waypoints[:, 0] += TURTLESIM_ORIGIN_X
#         waypoints[:, 1] += TURTLESIM_ORIGIN_Y
#         self._traj_waypoints  = waypoints
#         self._traj_idx        = 0
#         self._traj_step_timer = 0.0

#     def _step_pid(self):
#         if self._current_pose is None or self.goal_x is None or self.goal_y is None:
#             return

#         pose = self._current_pose

#         # --- goal reached (always checked against the final goal) ---
#         dx_goal = self.goal_x - pose.x
#         dy_goal = self.goal_y - pose.y
#         dist_to_goal = math.sqrt(dx_goal * dx_goal + dy_goal * dy_goal)

#         if dist_to_goal < self.goal_tolerance:
#             self._publish(0.0, 0.0)
#             self._phase = self._DONE
#             self.get_logger().info(
#                 f'[PID] Goal reached!  pos=({pose.x:.3f}, {pose.y:.3f})  '
#                 f'error={dist_to_goal:.4f} m  — waiting for next /goal_position'
#             )
#             return

#         # --- rotate-first phase: align toward final goal (CFM is translation-only) ---
#         if self._rotating:
#             angle_to_goal = math.atan2(dy_goal, dx_goal)
#             angle_error   = angle_to_goal - pose.theta
#             angle_error   = (angle_error + math.pi) % (2 * math.pi) - math.pi
#             w = self.pid_angular.compute(angle_error, self.dt)
#             self._publish(0.0, w)
#             if abs(angle_error) < self.angle_tolerance:
#                 self._rotating = False
#                 self.pid_linear.reset()
#                 self.pid_angular.reset()
#                 self.get_logger().info('[PID] Rotation done — starting CFM trajectory tracking')
#             return

#         # --- translation: follow CFM waypoints at 25 Hz (CFM_DT per waypoint) ---
#         if self._traj_waypoints is not None:
#             self._traj_step_timer += self.dt
#             if self._traj_step_timer >= CFM_DT:
#                 self._traj_step_timer = 0.0
#                 self._traj_idx += 1
#                 self.get_logger().debug(f'[PID] Waypoint {self._traj_idx}/{ORIGINAL_LEN}')
#                 if self._traj_idx >= ORIGINAL_LEN:
#                     # Trajectory exhausted — regenerate from current pose
#                     self._resample_trajectory(pose, label='re-sample')

#             target_x = self._traj_waypoints[self._traj_idx, 0]
#             target_y = self._traj_waypoints[self._traj_idx, 1]
#         else:
#             target_x, target_y = self.goal_x, self.goal_y

#         dx = target_x - pose.x
#         dy = target_y - pose.y
#         dist = math.sqrt(dx * dx + dy * dy)

#         # --- compute heading error toward current waypoint ---
#         angle_to_target = math.atan2(dy, dx)
#         angle_error     = angle_to_target - pose.theta
#         angle_error     = (angle_error + math.pi) % (2 * math.pi) - math.pi

#         # --- PID outputs ---
#         v = self.pid_linear.compute(dist, self.dt)
#         w = self.pid_angular.compute(angle_error, self.dt)

#         # Scale down linear speed when heading is far off
#         heading_factor = max(0.0, math.cos(angle_error))
#         v *= heading_factor

#         self._publish(v, w)

#     # ── helpers ───────────────────────────────────────────────────────────
#     def _publish(self, linear: float, angular: float = 0.0):
#         msg = Twist()
#         msg.linear.x  = float(linear)
#         msg.angular.z = float(angular)
#         self.cmd_vel_pub.publish(msg)


# # ── entry point ──────────────────────────────────────────────────────────────
# def main(args=None):
#     rclpy.init(args=args)
#     node = TurtleController()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()


# if __name__ == '__main__':
#     main()
