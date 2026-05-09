# Pose Trajectory Dataset

2D pose trajectory dataset for conditional trajectory generation using T-CFM (Trajectory Conditional Flow Matching).

## Dataset Summary

| Property | Value |
|----------|-------|
| File | `pose_traject_dataset_2000x50_0.npz` |
| Total samples | 100,000 |
| Trajectory length | 50 timesteps |
| State space | 2D plane (x, y) |
| File size | ~16 MB (compressed) |

## File Format

The dataset is stored as a NumPy `.npz` archive with three arrays:

```python
data = np.load("pose_traject_dataset_2000x50_0.npz", allow_pickle=True)
data['features']       # (100000, 14) — per-trajectory context features
data['targets']        # (100000, 50, 7) — trajectory sequences
data['feature_names']  # (14,) — column names for features
```

### Loading

```python
import numpy as np
import pandas as pd

data = np.load("pose_traject_dataset_2000x50_0.npz", allow_pickle=True)
df = pd.DataFrame(data['features'], columns=data['feature_names'])
df['targets'] = [t for t in data['targets']]
```

## Features (Context) — `features` array

Each sample has 14 context features describing the trajectory's physical constraints:

| Column | Name | Range | Description |
|--------|------|-------|-------------|
| 0 | `s_goal_x` | [-4.93, 4.99] | Goal position x |
| 1 | `s_goal_y` | [-4.99, 4.92] | Goal position y |
| 2 | `s_goal_z` | 0 (constant) | Goal position z (unused, 2D dataset) |
| 3 | `v_const` | [0.10, 0.20] | Constant velocity magnitude |
| 4 | `accel` | [0.02, 0.04] | Acceleration magnitude |
| 5 | `qdot_init_x` | [-0.2, 0.2] | Initial velocity x |
| 6 | `qdot_init_y` | [-0.2, 0.2] | Initial velocity y |
| 7 | `q_init_x` | [-4.91, 4.99] | Initial position x |
| 6 | `q_init_y` | [-4.99, 4.92] | Initial position y |
| 7 | `q_init_z` | 0 (constant) | Initial position z (unused) |
| 8 | `q_init_quatw` | [0.0004, 1.0] | Initial orientation quaternion w |
| 9 | `q_init_quatx` | 0 (constant) | Initial orientation quaternion x (unused) |
| 10 | `q_init_quaty` | 0 (constant) | Initial orientation quaternion y (unused) |
| 11 | `q_init_quatz` | [-1.0, 1.0] | Initial orientation quaternion z |
| 12 | `t_init` | [0.0001, 53.5] | Initial time within the full motion |
| 13 | `part_enum` | {0, 1, 2} | Motion phase (see below) |

<!-- ### Motion Phase (`part_enum`)

| Value | Phase | Count | Percentage |
|-------|-------|-------|------------|
| 0 | Acceleration | 19,993 | 20.0% |
| 1 | Constant velocity | 60,126 | 60.1% |
| 2 | Deceleration | 19,881 | 19.9% | -->

## Targets (Trajectories) — `targets` array

Each trajectory is a sequence of 50 timesteps with 7 dimensions:

| Column | Range | Description |
|--------|-------|-------------|
| 0 | [-4.93, 4.99] | **x position** |
| 1 | [-4.99, 4.92] | **y position** |
| 2 | 0 (constant) | z position (unused) |
| 3 | [0.0004, 1.0] | orientation quaternion w |
| 4 | 0 (constant) | orientation quaternion x (unused) |
| 5 | 0 (constant) | orientation quaternion y (unused) |
| 6 | [-1.0, 1.0] | orientation quaternion z |

For the T-CFM model, only columns 0-1 (x, y) are used as the generation target.

## How It's Used in Training

The training script (`scripts/train.py`) processes this dataset into two tensors:

### Context Vector C (model conditioning)
```
Shape: (N, 8)
Columns: [s_goal_x, s_goal_y, qdot_init_x, qdot_init_y, v_const, accel, q_init_x, q_init_y]

```

Constructed by:
1. Selecting `s_goal_x`, `s_goal_y`,`qdot_init_x`, `qdot_init_y`, `v_const`, `accel` from features
2. One-hot encoding `part_enum` into 3 binary columns
3. Selecting `q_init_x`, `q_init_y` from features

### Action Vector A (generation target)
```
Shape: (N, 64, 2)
Columns: [x_position, y_position] over 64 timesteps (padded from 50)
```

Constructed by:
1. Extracting columns 0-1 from targets → shape (N, 50, 2)
2. Padding with edge values from 50 to 64 timesteps (required by UNet architecture)

## T-CFM Model Architecture

The model generates 2D pose trajectories conditioned on physical context using **Trajectory Conditional Flow Matching (T-CFM)**. Flow matching learns a vector field that transforms random Gaussian noise into realistic trajectories in a single pass through an ODE.

### Inputs and Dimensions

```
Context Vector C ─── (B, 8) ──── "What trajectory should look like"
   [s_goal_x, s_goal_y, v_const, accel, part_0, part_1, part_2, q_init_x, q_init_y]
    ────────────────  ──────────  ─────  ───────────────────────  ─────────────────
      goal position   velocity    accel   one-hot motion phase     initial position
        (2 dims)      (1 dim)    (1 dim)      (3 dims)                (2 dims)

Noisy Trajectory ─── (B, 64, 2) ── "Starting point: pure Gaussian noise"
                      │    │  └── 2 channels: x, y position
                      │    └── 64 timesteps (padded from 50 for UNet compatibility)
                      └── batch size
```

### Architecture: 1D Convolutional UNet with FiLM Conditioning

```
                    Input: noisy trajectory x_t
                    Shape: (B, 64, 2)
                           │
                    rearrange to (B, 2, 64)   ← channels-first for Conv1d
                           │
┌──────────────────────────┼──────────────────────────────────┐
│                     CONDITIONING                            │
│                                                             │
│  Flow time t ──► SinusoidalPosEmb ──► MLP ──► (B, 32)     │
│                                                  │          │
│  Context C (B, 8) ───────────────────────────────┤          │
│                                              concat         │
│                                                  │          │
│                                           (B, 41) = global_feature
│                                           32 + 8 = 41      │
└──────────────────────────┼──────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  DOWN PATH  │
                    ├─────────────┤
                    │ Block: 2→32 │──────────────────────┐ skip
                    │ (B, 32, 64) │                      │
                    │ Downsample  │                      │
                    ├─────────────┤                      │
                    │ Block:32→128│──────────────┐ skip  │
                    │ (B,128, 32) │              │       │
                    │ Downsample  │              │       │
                    ├─────────────┤              │       │
                    │Block:128→256│──────┐ skip  │       │
                    │ (B,256, 16) │      │       │       │
                    ├─────────────┤      │       │       │
                    │  MID BLOCKS │      │       │       │
                    │ (B,256, 16) │      │       │       │
                    ├─────────────┤      │       │       │
                    │   UP PATH   │      │       │       │
                    ├─────────────┤      │       │       │
                    │Block:512→128│◄─────┘ cat   │       │
                    │ (B,128, 16) │              │       │
                    │  Upsample   │              │       │
                    ├─────────────┤              │       │
                    │Block:256→32 │◄─────────────┘ cat   │
                    │ (B, 32, 32) │                      │
                    │  Upsample   │                      │
                    ├─────────────┤                      │
                    │Block: 64→32 │◄─────────────────────┘ cat
                    │ (B, 32, 64) │
                    ├─────────────┤
                    │ Final Conv  │
                    │  32 → 2     │
                    └──────┬──────┘
                           │
                    rearrange to (B, 64, 2)
                           │
                    Output: predicted vector field v_θ
                    Shape: (B, 64, 2)
```

**Key details:**
- `dim_mults=(1, 4, 8)` → channel sizes `[32, 128, 256]` through the UNet
- 2 downsample/upsample layers → horizon must be divisible by 4 (hence padding 50→64)
- Total parameters: ~2.35M
- Each `ConditionalResidualBlock1D` uses **FiLM conditioning**: the 41-dim global feature (time embedding + context) modulates every conv layer via learned additive bias

### FiLM Conditioning (inside every block)

```
x ──► Conv1d ──► GroupNorm ──► Mish ──►  + bias  ──► Conv1d ──► + residual ──► out
                                          ▲
                        global_feature ──► Linear ──► bias (B, channels, 1)
                        (B, 41)
```

The context vector C is injected into every layer of the network, allowing the model to generate trajectories that respect the physical constraints at every scale.

### Training: Flow Matching Objective

From the T-CFM paper (Algorithm 1), the training minimizes the flow matching loss:

```python
x1 = real_trajectory                          # (B, 64, 2) — ground truth from dataset
x0 = torch.randn_like(x1)                    # (B, 64, 2) — pure Gaussian noise

t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
#   t  = random time in [0, 1]               — scalar per sample
#   xt = t * x1 + (1-t) * x0                 — linearly interpolated trajectory
#   ut = x1 - x0                             — target vector field (straight flow)

vt = model(t, xt, global_cond=context)        # model predicts vector field
loss = mean((vt - ut)^2)                      # match the true flow direction
```

The model learns: "Given a noisy trajectory `xt` at flow time `t` with context `C`, what direction should I push it toward real data?"

### Sampling (Inference): Euler ODE Integration

At inference, the model generates trajectories by integrating the learned vector field from noise to data:

```
Start:  x_0 ~ N(0, I)                          shape: (B, 64, 2)  ← random noise

Step 1: x_{1/N}   = x_0     + (1/N) * v_θ(0/N,     x_0,     C)
Step 2: x_{2/N}   = x_{1/N} + (1/N) * v_θ(1/N,   x_{1/N},   C)
  ...
Step N: x_1       = x_{N-1/N} + (1/N) * v_θ((N-1)/N, x_{N-1/N}, C)

End:    x_1 = generated trajectory              shape: (B, 64, 2)  ← realistic trajectory
        trimmed to (B, 50, 2)                   ← remove padding
```

Each step asks the model "which direction?" and takes a small step. After N steps (default 100), noise becomes a trajectory. A key advantage of flow matching: **even N=1 produces reasonable results**, enabling 100x speedup over diffusion models.

### Output

```
Generated Trajectory ─── (B, 50, 2)
                          │    │  └── x, y position at each timestep
                          │    └── 50 timesteps (padding removed)
                          └── batch size
```

### Dimension Summary

| Component | Shape | Description |
|-----------|-------|-------------|
| Input noise | `(B, 64, 2)` | Random Gaussian, starting point for generation |
| Context C | `(B, 8)` | Physical constraints guiding generation |
| Flow time t | `(B,)` | Scalar in [0,1], sinusoidal embedding → 32-dim |
| Global feature | `(B, 41)` | Concatenation of time embedding (32) + context (8) |
| Model output | `(B, 64, 2)` | Predicted vector field (velocity in trajectory space) |
| Final output | `(B, 50, 2)` | Generated x,y trajectory (trimmed from 64) |

## Related Files

| File | Description |
|------|-------------|
| `pose_traject_dataset_2000x50_0.npz` | Compressed dataset (single file) |
| `pose_traject_dataset_2000x50_0/` | Extracted numpy arrays (features, targets, feature_names) |
| `athit_dataset.csv` | Raw CSV with columns: `position_x_m`, `position_y_m`, `goal_x_m`, `goal_y_m`, `v_max`, `a_max`, `window_phase`, `window_uid` |
