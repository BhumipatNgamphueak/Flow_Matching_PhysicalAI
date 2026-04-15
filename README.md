# Flow Matching – Physical AI (TurtleSim)

> **Repository:** https://github.com/BhumipatNgamphueak/Flow_Matching_PhysicalAI.git  
> **Branch:** `TurtleSim`

A ROS 2 workspace for experimenting with motion-control algorithms on an enhanced TurtleSim.  
The `trajectory_publisher` package is the **contribution layer** — add your trajectory node here and it plugs straight into the simulator.

---

## Quick Start

```bash
git clone -b TurtleSim https://github.com/BhumipatNgamphueak/Flow_Matching_PhysicalAI.git
cd Flow_Matching_PhysicalAI

sudo apt install -y python3-matplotlib python3-pygame ros-humble-turtlesim
rosdep install --from-paths src --ignore-src -r -y

colcon build --symlink-install
source install/setup.bash

# default (trapezoidal velocity)
ros2 launch trajectory_publisher turtlesim_trapezoid.launch.py

# PID position control
ros2 launch trajectory_publisher turtlesim_trapezoid.launch.py mode:=pid goal_x:=12.0 goal_y:=10.0
```

---

## World

| Property | Value |
|----------|-------|
| Size | 15 m × 15 m |
| Default turtle spawn | (7.5, 7.5) — centre |
| θ = 0 heading | +X direction |
| Boundary | clamps at 0 and 15 m |
| Simulator rate | 100 Hz |

---

## Interface — the only two things you need to know

### Velocity control → publish here

```
Topic : /turtle1/cmd_vel
Type  : geometry_msgs/Twist

Twist.linear.x   = forward speed  (m/s)
Twist.angular.z  = turn rate      (rad/s,  + = counter-clockwise)
```

### Position control → read pose from here, then publish cmd_vel

```
Topic : /turtle1/pose
Type  : turtlesim/Pose

Pose.x                # current X  (m)
Pose.y                # current Y  (m)
Pose.theta            # heading    (rad)
Pose.linear_velocity  # actual linear speed  (m/s)
Pose.angular_velocity # actual angular speed (rad/s)
```

Your controller reads `/turtle1/pose` → computes error → publishes `/turtle1/cmd_vel`.

---

## Running the Controller

All motion is handled by **`turtle_controller.py`**, which has two modes selectable at launch time.

---

### Mode 1 — Velocity control (Trapezoidal profile)

The turtle drives straight along the +X axis using a pre-computed trapezoidal velocity profile.  
No feedback is used — velocity is commanded open-loop.

```bash
ros2 launch trajectory_publisher turtlesim_trapezoid.launch.py mode:=trapezoid
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `v_max` | `2.0` m/s | Peak velocity |
| `a_max` | `0.5` m/s² | Acceleration / deceleration rate |
| `distance` | `5.0` m | Total straight-line distance to travel |
| `start_delay` | `2.0` s | Wait time before motion starts |

```bash
# Example — slow ramp over 8 m
ros2 launch trajectory_publisher turtlesim_trapezoid.launch.py \
    mode:=trapezoid v_max:=1.5 a_max:=0.3 distance:=8.0
```

**Profile shape:**
```
velocity
  ^
  |     ___________
  |    /           \
  |   /             \
  |  /               \
  |_/                 \__
  +───────────────────────▶ time
    accel | constant | decel
```

If `distance` is too short to reach `v_max`, it automatically falls back to a triangular profile.

---

### Mode 2 — Position control (PID)

The turtle drives to a goal position `(goal_x, goal_y)` using two PID loops.  
Pose feedback comes from `/turtle1/pose`.

```bash
ros2 launch trajectory_publisher turtlesim_trapezoid.launch.py mode:=pid
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `goal_x` | `10.0` m | Target X position |
| `goal_y` | `7.5` m | Target Y position |
| `goal_tolerance` | `0.05` m | Distance at which the turtle stops |
| `kp_linear` | `1.5` | Linear PID — proportional gain |
| `ki_linear` | `0.0` | Linear PID — integral gain |
| `kd_linear` | `0.05` | Linear PID — derivative gain |
| `kp_angular` | `5.0` | Angular PID — proportional gain |
| `ki_angular` | `0.0` | Angular PID — integral gain |
| `kd_angular` | `0.1` | Angular PID — derivative gain |
| `v_max_pid` | `2.0` m/s | Max linear speed |
| `w_max_pid` | `3.0` rad/s | Max angular speed |

```bash
# Example — drive to (12, 10) with stronger angular correction
ros2 launch trajectory_publisher turtlesim_trapezoid.launch.py \
    mode:=pid goal_x:=12.0 goal_y:=10.0 kp_angular:=6.0
```

**Control diagram:**
```
/turtle1/pose ──► [ error compute ] ──► [ PID linear  ] ──► linear.x  ──► /turtle1/cmd_vel
                                    └──► [ PID angular ] ──► angular.z ──►
```

---

## Plotter

Runs automatically on every launch.  
Subscribes to `/turtle1/pose` + `/turtle1/cmd_vel` → live 4-panel matplotlib window.  
Plot saved to `~/turtle_trajectory.png` on Ctrl-C.

---

## Other Useful Topics & Services

```
/spawn_turtle      turtlesim/Spawn     — spawn extra turtle at (x, y, θ)
/remove_turtle     turtlesim/Kill      — remove turtle by name
/spawn_pizza       GivePosition        — place a pizza at (x, y)
/turtle1/stop      std_srvs/Empty      — zero velocity immediately
/turtle1/scan      ScannerDataArray    — nearby entities (type, angle, distance)
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `executable not found` | `chmod +x scripts/your_node.py` |
| Build fails (symlink error) | `rm -rf build/ install/ log/` then rebuild |
| Turtle doesn't move | `start_delay` defaults to 2 s — wait for it |
| Turtle hits wall | World is 0–15 m; check your goal / distance values |
| Blank matplotlib window | `sudo apt install python3-pyqt5`, change backend to `Qt5Agg` |
