# Flow Matching – Physical AI (TurtleSim)

> **Repository:** https://github.com/BhumipatNgamphueak/Flow_Matching_PhysicalAI
> **Branch:** `TurtleSim`

Compare classical **trapezoidal velocity profiles** (computed via equation) against
**Conditional Flow Matching (CFM)** — a generative model that learns to produce
trajectories — under identical constraints issued by a Gemini-based LLM planner.

```
              ┌─────────────────┐
   "go to     │  LlmPlanner     │   /traj_context
    x=2,y=1   │  (Gemini 2.5)   │ ────────────────┐
    v=0.15"   │  /LlmPrompt     │                 │
              └─────────────────┘                 ▼
                                       ┌─────────────────────┐
                                       │  turtle_controller  │
                                       │  ┌──────────────┐   │   /turtle1/cmd_vel
                                       │  │ trapezoid OR │   │ ────────────────┐
                                       │  │ CFM + PID    │   │                 │
                                       │  └──────────────┘   │                 ▼
                                       └─────────────────────┘     ┌──────────────────┐
                                                                   │  turtlesim_plus  │
                                                                   │   simulator      │
                                                                   └──────────────────┘

   Comparison logs → ~/comparison_logs/{timestamp}_pid.npz   (CFM + analytical classical)
```

---

## Requirements

| | Tested with |
|---|---|
| OS | Ubuntu 22.04 LTS |
| ROS | ROS 2 Humble |
| Python | 3.10 |
| GPU (optional) | NVIDIA with CUDA 12.x driver — falls back to CPU |
| LLM API | [Google AI Studio key](https://aistudio.google.com/apikey) (free tier works) |

Disk needed during install: ~6 GB (CUDA torch is ~3.5 GB).

---

## Install

### 1. System packages

```bash
sudo apt update && sudo apt install -y \
    python3-pip \
    python3-pygame \
    ros-humble-turtlesim
```

### 2. Clone the workspace + the LLM planner

The LLM planner lives in a separate repo under `src/`:

```bash
git clone -b TurtleSim https://github.com/BhumipatNgamphueak/Flow_Matching_PhysicalAI.git
cd Flow_Matching_PhysicalAI

git clone https://github.com/lyDevper/LlmPlanner-ROS2.git src/LlmPlanner-ROS2
```

### 3. Make Python node scripts executable

ROS 2 launch refuses to start a Python node without `+x`. The exact set of nodes
launched by the system:

```bash
chmod +x \
    src/trajectory_publisher/scripts/turtle_controller.py \
    src/trajectory_publisher/scripts/turtle_plotter.py \
    src/turtlesim_plus/turtlesim_plus/scripts/turtlesim_plus_node.py \
    src/LlmPlanner-ROS2/LlmPlanner/src/llm_pack/scripts/llm_node.py
```

(Or just bulk-mark everything under `scripts/` as executable — harmless:
`find src -path "*/scripts/*.py" -exec chmod +x {} \;`)

### 4. Install Python dependencies

```bash
# numpy MUST be < 2 — Ubuntu's apt-installed matplotlib was built against numpy 1.x
pip install --user "numpy<2"

# CFM runtime deps (small, pure-python)
pip install --user einops torchdiffeq torchsummary pot

# torchcfm + torchdyn would otherwise drag in a fresh CUDA torch — install with --no-deps
pip install --user --no-deps torchcfm torchdyn

# LangChain + Gemini for the LLM planner
pip install --user -r src/LlmPlanner-ROS2/LlmPlanner/requirements.txt
```

### 5. Install PyTorch — pick ONE

```bash
# Option A — NVIDIA GPU with CUDA 12.x driver (recommended; ~3.5 GB)
pip install --user torch --index-url https://download.pytorch.org/whl/cu121

# Option B — CPU only (~200 MB; CFM is small enough for CPU inference)
pip install --user torch --index-url https://download.pytorch.org/whl/cpu
```

If you run out of disk during install: `pip cache purge` and retry.

### 6. Build the workspace

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

`--symlink-install` is important: edits to source `.py` files apply without rebuilding.

---

## Configure the LLM API key

Get a free key at https://aistudio.google.com/apikey, then:

```bash
echo 'GOOGLE_API_KEY="<paste-your-key>"' > .env
chmod 600 .env
```

`.env` is gitignored — it will not leak when you push.

---

## Run

Two terminals — both at `~/Flow_Matching_PhysicalAI`.

### Terminal 1 — launch the system

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
set -a; source .env; set +a   # exports GOOGLE_API_KEY for the launch

# CFM + PID mode (the comparison run — CFM drives, classical baseline is logged)
ros2 launch trajectory_publisher turtlesim_trapezoid.launch.py mode:=pid

# Or classical-only mode (no CFM model loaded; faster startup)
ros2 launch trajectory_publisher turtlesim_trapezoid.launch.py mode:=trapezoid
```

What you should see on success:
```
[turtle_controller] Subscribed to /traj_context (LLM constraints)
[turtle_controller] [PID] Loading CFM 3D model …
[turtle_controller] [PID] CFM model ready.
[llm_node]          LlmNode initialized. Listening on /LlmPrompt, publishing to /traj_context.
```

### Terminal 2 — issue an LLM prompt

```bash
source install/setup.bash

ros2 service call /LlmPrompt llm_pack_interface/srv/String \
    "{prompt: 'Move to x=2.0, y=1.5 with v=0.15 and a=0.03'}"
```

The launch terminal will then log:
```
[llm_node]          Agent reply: ...
[turtle_controller] [TrajContext] v_const=0.1500 a=0.0300 goal=(9.500, 9.000) world
[turtle_controller] [CFM-3D] Trajectory LLM-context in 8.4 ms (v_const=0.1500, a=0.0300)
[turtle_controller] [PID] LLM goal applied: (9.500, 9.000); CFM resampled
[turtle_controller] [log] Saved comparison → /home/<user>/comparison_logs/...
```

The turtle drives toward the goal at ~0.15 m/s.

---

## Constraint ranges (LLM is prompted to stay within these)

| Field | Range | Source |
|---|---|---|
| `v_const` (linear velocity) | 0.10 – 0.20 m/s | LlmPlanner system prompt |
| `a` (linear acceleration) | 0.02 – 0.04 m/s² | LlmPlanner system prompt |
| Goal `(x, y, z)` | meters, relative to current turtle pose | LLM tool input |

The LLM treats goals as displacements from the turtle's *current* pose at the moment
the prompt arrives ("robot starts at (0,0,0)"). The controller adds the live pose to
get world coordinates.

---

## Modes

### `mode=trapezoid` — classical baseline
Open-loop trapezoidal velocity profile. Drives along the current heading for the
LLM-issued distance with peak velocity `v_const` and acceleration `a`. CFM model
is **not** loaded.

### `mode=pid` — CFM + PID (the headline mode)
1. Loads a Conditional Flow Matching model (~2.35 M params) at startup.
2. On every `/traj_context` arrival, builds a 13-dim context vector
   `[goal_xy, goal_θ, v_const, a, ω, α, q_init_xyθ, qdot_xyθ]`
   and samples a (50, 3) waypoint trajectory in ~5–10 ms (10-step Euler ODE).
3. Tracks waypoints with a PID controller that publishes `/turtle1/cmd_vel`.
4. Logs **both** the CFM trajectory and an analytical classical baseline (under
   identical constraints, toward-goal heading) to `~/comparison_logs/`.

---

## Manual goal (no LLM)

If you don't want to use the LLM, publish directly:

```bash
ros2 topic pub /goal_position geometry_msgs/Point "{x: 10.0, y: 7.5, z: 0.0}"
```

The PID/CFM mode reuses cached LLM constraints if any have arrived; otherwise it
falls back to `V_CONST_FALLBACK = 0.156 m/s` and `A_FALLBACK = 0.028 m/s²`.

---

## Comparison logs

```
~/comparison_logs/{YYYYMMDD-HHMMSS-mmm}_pid.npz
  cfm_waypoints       (50, 3)  CFM-generated x, y, θ in world coords
  classical_waypoints (N, 3)   Analytical trapezoid baseline toward same goal
  cfm_dt              0.04     seconds per CFM waypoint
  classical_dt        0.04     seconds per classical waypoint
  v_const, a                   constraints used
  goal_world          (2,)     target in world coords
  start_pose          (3,)     pose at sample time
```

Quick plot:

```python
import numpy as np, matplotlib.pyplot as plt
import glob
path = sorted(glob.glob("~/comparison_logs/*_pid.npz"))[-1]
d = np.load(path)
plt.plot(*d['cfm_waypoints'][:, :2].T, label='CFM')
plt.plot(*d['classical_waypoints'][:, :2].T, label='Classical')
plt.axis('equal'); plt.legend(); plt.show()
```

---

## World

| Property | Value |
|---|---|
| Size | 0 – 15 m on each axis |
| Default spawn | (7.5, 7.5), θ = 0 (+X) |
| θ convention | radians, + = counter-clockwise |
| Boundary | hard clamp at 0 and 15 |
| Simulator rate | 100 Hz |

> **Note** — the controller uses `TURTLESIM_ORIGIN_X/Y = 5.544` to convert between
> world frame and the meter-frame the CFM model was trained on. This 5.544 reflects
> the *training-data normalization*, not the current 7.5 spawn point. Goals issued
> via `/traj_context` use the live pose as origin (not 5.544), so this is fine for
> LLM-driven runs.

---

## Project layout

```
Flow_Matching_PhysicalAI/
├── src/
│   ├── trajectory_publisher/                   # Main ROS 2 package
│   │   ├── launch/turtlesim_trapezoid.launch.py
│   │   ├── scripts/
│   │   │   ├── turtle_controller.py            # Active controller
│   │   │   ├── turtle_plotter.py               # Live matplotlib plot
│   │   │   ├── diffuser/                       # CFM model + datasets + utils
│   │   │   ├── config/cfm_pose.py              # Training hyperparameters
│   │   │   ├── evaluate/                       # Offline evaluation
│   │   │   └── logs/.../state_192000.pt        # Trained checkpoint
│   │   └── package.xml                         # Declares llm_pack_interface dep
│   ├── turtlesim_plus/                         # Enhanced 2D simulator
│   └── LlmPlanner-ROS2/                        # Gemini-based constraint planner
│       └── LlmPlanner/src/llm_pack/scripts/llm_node.py
├── .env                                        # GOOGLE_API_KEY (gitignored)
├── .gitignore
└── README.md
```

---

## Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `executable 'llm_node.py' not found on the libexec directory` | Source script lacks `+x` bit. Run the chmod step in §3 of Install. |
| `ModuleNotFoundError: No module named 'torch'` | Install torch via pip (Install §5). |
| `ModuleNotFoundError: No module named 'einops'` (or torchcfm, pot, …) | Re-run Install §4. |
| `ImportError: cannot import name 'create_agent' from 'langchain.agents'` | Outdated `llm_node.py`. The active code only needs `create_react_agent` from langgraph. |
| `numpy.core.multiarray failed to import` (in `import matplotlib`) | numpy ≥ 2.0 vs system matplotlib (built for numpy 1.x). `pip install --user "numpy<2"`. |
| `[WARN] [llm_node]: GOOGLE_API_KEY not set!` | `.env` not sourced. `set -a; source .env; set +a`. |
| `The passed service type is invalid` | Workspace not sourced in this terminal. `source install/setup.bash`. |
| Service `/LlmPrompt` waiting forever | `llm_node` died. Check launch output for the actual stack trace. |
| `FileNotFoundError: state_192000.pt` | Checkpoint missing. Verify `src/trajectory_publisher/scripts/logs/pose_trajectory_3DV2/cfm/H64_T100/20260505-1157/state_192000.pt` exists. |
| Disk full during `pip install ... cu121` | Run `pip cache purge`, free up space, retry. CUDA torch is ~3.5 GB. |
| `Connection broken: ... No space left on device` | Same as above. |
| Build fails with "existing path cannot be removed" symlink error | Stale half-build. `rm -rf build/<pkg> install/<pkg>` and rebuild. |
| Turtle doesn't move after service call | Check launch terminal: did `[CFM-3D] Trajectory ...` log appear? If yes, the turtle IS moving — at 0.15 m/s × 2.5 m it takes ~17 s. |
| Blank matplotlib window | `sudo apt install python3-pyqt5` and set `MPLBACKEND=Qt5Agg`. |
| Multiple installs of `llm_pack` conflict | Remove any nested `install/` dirs (e.g. `src/LlmPlanner-ROS2/LlmPlanner/install`). |

---

## Backup plan: classical-only manual mode

If the LLM stack is broken and you just want to drive the turtle:

```bash
ros2 launch trajectory_publisher turtlesim_trapezoid.launch.py \
    mode:=trapezoid v_max:=1.5 a_max:=0.3 distance:=8.0
```

Or in PID mode with manual goal:

```bash
ros2 launch trajectory_publisher turtlesim_trapezoid.launch.py mode:=pid use_llm:=false
ros2 topic pub /goal_position geometry_msgs/Point "{x: 10.0, y: 7.5, z: 0.0}"
```

`use_llm:=false` skips spawning `llm_node` (no Google key required).
