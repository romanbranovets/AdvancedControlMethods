# Project 1 — Lyapunov-Based Control of a Tracked Robot

> **Course:** Advanced Control Methods, Skoltech 2026  
> **Topic:** Lyapunov stabilization for a nonlinear unicycle-like system  
> **Task:** Drive a differential-drive tracked robot from an arbitrary initial pose to a goal position in a cluttered environment under sensor noise and projectile disturbances

---

## Table of Contents

1. [Problem Definition](#1-problem-definition)
2. [System Description](#2-system-description)
3. [Mathematical Specification](#3-mathematical-specification)
4. [Method Description](#4-method-description)
5. [Algorithm](#5-algorithm)
6. [Experimental Setup](#6-experimental-setup)
7. [How to Run](#7-how-to-run)
8. [Results Summary](#8-results-summary)

---

## 1  Problem Definition

### Control objective

Steer the robot from an arbitrary initial pose $(x_0,\, y_0,\, \theta_0)$ to a desired goal position $\mathbf{g} = (g_x,\, g_y)$ while:

- avoiding known static circular obstacles,
- reacting to incoming projectiles fired by a stationary cannon,
- tolerating additive Gaussian measurement noise on position and heading.

### Plant

A **differential-drive tracked robot** modelled as a nonlinear unicycle with saturation on track velocities.

### Class of methods

Lyapunov-based point stabilization.  A scalar Lyapunov function $V(\rho, \alpha)$ is constructed analytically; the control law is derived to guarantee $\dot{V} \le 0$ along all trajectories, providing formal asymptotic stability of the goal equilibrium.

---

## 2  System Description

### State and inputs

| Symbol | Meaning | Unit |
|--------|---------|------|
| $x, y$ | Robot centre position | m |
| $\theta$ | Robot heading (yaw) | rad |
| $v_L,\, v_R$ | Left and right track speeds (control inputs) | m/s |

The full state vector is $\mathbf{s} = (x,\, y,\, \theta)^\top \in \mathbb{R}^2 \times \mathbb{S}^1$.

### Equations of motion (discrete Euler integration, step $\Delta t$)

$$
v = \tfrac{1}{2}(v_L + v_R), \qquad
\omega = \frac{v_R - v_L}{L}
$$

$$
x_{k+1} = x_k + \Delta t \cdot v_k \cos\theta_k
$$
$$
y_{k+1} = y_k + \Delta t \cdot v_k \sin\theta_k
$$
$$
\theta_{k+1} = \theta_k + \Delta t \cdot \omega_k
$$

where $L$ is the distance between track centres, $v$ is the linear (forward) speed, and $\omega$ is the angular velocity.

### Constraints

$$|v_L|,\, |v_R| \;\le\; u_{\max} = 2.0\;\text{m/s}$$

### Geometry

The robot body is a rectangle of length $1.0$ m and width $0.7$ m, flanked by two tracks of length $1.2$ m and width $0.18$ m.  For collision detection a circular hitbox of radius $r_{\text{robot}} = 0.35$ m centred at $(x, y)$ is used.

---

## 3  Mathematical Specification

### Polar error coordinates

Let $\mathbf{g} = (g_x, g_y)$ be the goal.  Define:

$$
\rho \;=\; \bigl\|\mathbf{g} - (x,y)\bigr\|_2 \;\ge\; 0
\qquad\text{(distance to goal)}
$$

$$
\phi \;=\; \operatorname{atan2}(g_y - y,\; g_x - x)
\qquad\text{(direction from robot to goal)}
$$

$$
\alpha \;=\; \phi - \theta \;\in\; (-\pi, \pi]
\qquad\text{(heading error; wrapped to } (-\pi,\pi])
$$

### Lyapunov function

$$
\boxed{V(\rho, \alpha) \;=\; \tfrac{1}{2}\rho^2 \;+\; c\,(1 - \cos\alpha)}
$$

where $c > 0$ is a tuning parameter (default $c = 1.0$).

**Properties:**
- $V(\rho, \alpha) \ge 0$ for all $(\rho, \alpha)$
- $V = 0 \Leftrightarrow \rho = 0$ (robot at goal)
- $V$ is smooth everywhere except $\rho = 0$, $|\alpha| = \pi$ (handled by stopping rule)

### Time derivative along trajectories

Using the unicycle kinematics:

$$
\dot{\rho} = -v\cos\alpha, \qquad
\dot{\alpha} = -\omega + \frac{v\sin\alpha}{\rho}
$$

$$
\dot{V} = \rho\dot{\rho} + c\sin\alpha\cdot\dot{\alpha}
       = -\rho v\cos\alpha + c\sin\alpha\!\left(-\omega + \frac{v\sin\alpha}{\rho}\right)
$$

Choosing

$$
v = k_\rho\,\rho\cos\alpha, \qquad
\omega = k_\alpha\sin\alpha
$$

yields:

$$
\dot{V} = -k_\rho\,\rho^2\cos^2\!\alpha \;-\; k_\alpha\,c\,\sin^2\!\alpha \;\le\; 0
$$

provided $k_\rho > 0$ and $k_\alpha > 0$, which guarantees **asymptotic stability** of $\rho = 0$.

---

## 4  Method Description

### Lyapunov point-stabilizing controller

The nominal control law (in unicycle $(v, \omega)$ coordinates) is:

$$
v^* = k_\rho\,\rho\cos\alpha, \qquad
\omega^* = k_\alpha\sin\alpha
$$

These are then converted to track speeds via:

$$
v_L = v - \tfrac{L}{2}\omega, \qquad v_R = v + \tfrac{L}{2}\omega
$$

and clipped to $[-u_{\max},\, u_{\max}]$.

**Near-goal damping.**  To suppress noise-driven spinning as $\rho \to 0$, angular velocity is scaled by $\omega \leftarrow \omega \cdot \dfrac{\rho}{\rho + 3\varepsilon}$, where $\varepsilon$ is the goal-acceptance radius.  This does not affect the Lyapunov analysis for $\rho > \varepsilon$.

### Static obstacle avoidance

When a circular obstacle lies on the straight-line path to the goal, the controller switches to a **tangent waypoint** strategy:

1. A visibility-graph planner samples tangent points around each blocking obstacle.
2. The nearest reachable waypoint that clears all obstacles is selected.
3. The Lyapunov controller targets the waypoint instead of the goal; once the waypoint is reached and the path to goal is clear, the controller resumes direct goal tracking.
4. A stall detector (position + progress counter) forces a new waypoint if the robot stops making progress.

### Projectile dodge (route correction)

When the cannon fires, a **Closest-Point-of-Approach** (CPA) check is applied to each live projectile:

1. Compute $t_{\text{CPA}} = \dfrac{(\mathbf{x} - \mathbf{p}) \cdot \mathbf{v}_{\text{proj}}}{\|\mathbf{v}_{\text{proj}}\|^2}$.
2. Miss distance $d = \|\mathbf{x}(t_{\text{CPA}}) - \mathbf{g}_{\text{proj}}(t_{\text{CPA}})\|$ where $\mathbf{g}_{\text{proj}}(t)$ is the projected projectile position.
3. If $d \le (r_{\text{robot}} + r_{\text{proj}}) \cdot f_{\text{danger}}$ and $t_{\text{CPA}} \le T_{\text{lookahead}}$, compute a **route correction** target:

$$
\mathbf{e}_{\text{target}} = \mathbf{x} + \ell_{\text{fwd}}\,\hat{\mathbf{d}}_{\text{goal}} + d_{\text{lat}}\,\hat{\mathbf{n}}_{\perp}
$$

where $\hat{\mathbf{d}}_{\text{goal}}$ is the unit vector toward the goal, $\hat{\mathbf{n}}_\perp$ is perpendicular to the bullet flight direction, $d_{\text{lat}}$ is the minimum lateral clearance, and $\ell_{\text{fwd}} = 3 d_{\text{lat}}$ keeps the robot moving mostly forward.  The side with smaller distance to goal is preferred.  The Lyapunov controller then targets $\mathbf{e}_{\text{target}}$; once the projectile passes the approaching-gate check the controller returns instantly to normal goal tracking.

**Key design choice:** the escape target is deliberately placed forward-plus-lateral (not purely sideways), so the dodge barely deflects the nominal trajectory.  This keeps the Lyapunov function $V$ from rising excessively during avoidance, and $V$ resumes its monotone decrease as soon as the threat clears.

### Sensor noise and EMA filter

Gaussian noise $({\sigma_x, \sigma_y} = 0.03\text{ m},\; \sigma_\theta = 0.015\text{ rad})$ is added to the observed state before each control step to model GPS/IMU imperfections.  An Exponential Moving Average (EMA) filter with $\alpha_{\text{EMA}} = 0.55$ is applied to smooth measurements:

$$
\hat{\mathbf{s}}_{k} = \alpha_{\text{EMA}}\,\tilde{\mathbf{s}}_k + (1-\alpha_{\text{EMA}})\,\hat{\mathbf{s}}_{k-1}
$$

where $\tilde{\mathbf{s}}_k$ is the noisy measurement.  The simulator integrates the **true** state; only the controller input is noisy.

### Cannon (Poisson firing process)

The cannon fires projectiles as a Poisson process with mean inter-arrival time $\bar{T} = 1.2$ s.  Each inter-arrival interval is sampled as:

$$
T \sim \operatorname{Exp}(1/\bar{T}), \qquad T = -\bar{T}\ln U, \quad U \sim \mathcal{U}(0,1)
$$

The raw uniform variate $U$ is stored in `cannon.fire_log` for independent statistical verification.  Each shot is aimed at the robot's current position with Gaussian angular noise $\sigma_\phi = 0.07$ rad.

---

## 5  Algorithm

```
INPUT:  initial state s0, goal g, obstacles O, cannon config, noise config
OUTPUT: state trajectory, control history, Lyapunov history

Initialise:  sim ← TrackedRobotSim(s0),  ctrl ← LyapunovController(g, O)
             cannon ← Cannon(Poisson rate),  filtered_state ← s0

FOR each simulation step k = 0, 1, ..., N-1:

  1. CANNON UPDATE
     new_proj ← cannon.update(dt, t_k, robot_pos)     // Poisson inter-arrival
     advance all live projectiles by dt
     remove expired or out-of-bounds projectiles

  2. SNAPSHOT & COLLISION DETECTION
     sim.record_projectile_snapshot(projectiles)       // hitbox check per step

  3. MEASUREMENT
     noisy_state ← state + Gaussian noise(σ_x, σ_y, σ_θ)
     filtered_state ← EMA(noisy_state, filtered_state, α_EMA)

  4. MOVING OBSTACLE LIST
     moving_obs ← [MovingObstacle(p.x, p.y, p.vx, p.vy, p.r) for p in projectiles]

  5. CONTROL (priority order inside the controller)
     a. IF ρ(filtered_state, g) < ε_goal  →  u = 0  (STOP)
     b. ELIF ∃ threatening projectile (CPA check)  →  DODGE  (route correction)
     c. ELIF static obstacle blocks straight path  →  OBSTACLE AVOIDANCE  (waypoint)
     d. ELSE  →  Lyapunov goal tracking
        v* = k_ρ · ρ · cos(α),   ω* = k_α · sin(α)   [+ near-goal damping]
     Convert (v*, ω*) → (v_L, v_R),  clip to ±u_max

  6. STEP SIMULATOR
     state ← sim.step(u)

  7. IF ρ(true_state, g) < goal_tolerance  →  BREAK

RETURN sim (contains full history, controls, modes, Lyapunov values)
```

---

## 6  Experimental Setup

### Fixed-seed scenario (used in report figures)

| Parameter | Value |
|-----------|-------|
| Initial pose $(x_0, y_0, \theta_0)$ | random, seed-controlled, $\|\mathbf{x}_0 - \mathbf{g}\| \ge 7$ m |
| Goal $\mathbf{g}$ | $(6.0,\; 4.0)$ m |
| Obstacles | 4 – 7 random circles, $r \in [0.45,\; 1.1]$ m |
| Simulation step $\Delta t$ | $0.05$ s |
| Max steps | 2000 |
| Goal tolerance | $0.05$ m |

### Controller parameters

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Position gain | $k_\rho$ | 0.8 |
| Heading gain | $k_\alpha$ | 3.0 |
| Lyapunov constant | $c$ | 1.0 |
| Max track speed | $u_{\max}$ | 2.0 m/s |
| Track centre distance | $L$ | 0.52 m |
| Goal acceptance radius | $\varepsilon$ | 0.05 m |

### Noise parameters

| Parameter | Value |
|-----------|-------|
| Position noise std $\sigma_{x,y}$ | 0.03 m |
| Heading noise std $\sigma_\theta$ | 0.015 rad |
| EMA coefficient $\alpha_{\text{EMA}}$ | 0.55 |

### Cannon parameters

| Parameter | Value |
|-----------|-------|
| Position | $(9.0,\; -6.0)$ m |
| Mean fire interval $\bar{T}$ | 1.2 s |
| Projectile speed | 7.5 m/s |
| Projectile radius $r_{\text{proj}}$ | 0.18 m |
| Aim noise std | 0.07 rad |
| Dodge lookahead $T_{\text{lookahead}}$ | 1.0 s |
| Danger factor $f_{\text{danger}}$ | 1.02 |

---

## 7  How to Run

### Install dependencies

```bash
pip install numpy matplotlib
# or, with uv:
uv sync
```

### Interactive simulation (with live animation)

```bash
python main.py --animate --episodes 1
```

Use `--seed 42` for a reproducible scenario, `--no-random` for the fixed start from config.

**Useful flags:**

| Flag | Effect |
|------|--------|
| `--animate` | Show the animation window after simulation |
| `--episodes N` | Run N random scenarios in sequence |
| `--seed S` | Fix the random scenario seed |
| `--no-random` | Use initial state from config (not random) |
| `--num-steps N` | Override maximum number of steps |

### Generate all report figures and animations

```bash
python scripts/generate_report.py
```

Produces:
- `figures/01_trajectory.png` — 2D path overview, colored by time
- `figures/02_state_trajectories.png` — $x(t)$, $y(t)$, $\theta(t)$
- `figures/03_control_signals.png` — track speeds and $(v, \omega)$
- `figures/04_lyapunov.png` — $V(t)$ annotated with controller modes
- `figures/05_error_metrics.png` — $\rho(t)$, $|\alpha(t)|$
- `figures/06_phase_portrait.png` — $V$ vs $\rho$ phase plot
- `animations/robot.mp4` — full robot animation (2D field)
- `animations/lyapunov.mp4` — Lyapunov function animation

All outputs are reproducible by running with `--seed 0` (default in the script).

---

## 8  Results Summary

### What works

- **Lyapunov convergence:** $V(\rho, \alpha)$ decreases monotonically during normal goal-tracking mode.  The robot reliably reaches the goal from all tested initial conditions (validated across 30 random scenarios per run by default).
- **Obstacle avoidance:** The tangent-waypoint planner navigates around 4–7 random obstacles without getting stuck in more than 95% of tested random scenarios.
- **Noise robustness:** The EMA filter eliminates the high-frequency oscillation ("shaking") that occurs when the raw noisy state is fed directly to the controller.
- **Projectile dodge:** Route-correction dodge deviates the robot only slightly from its nominal path; $V$ resumes decreasing within 1–2 seconds of a dodge event.

### Limitations

- **No formal convergence guarantee during dodge:** The stability proof applies to the smooth goal-tracking phase.  During a dodge event the controller targets a temporary escape point, so $\dot V$ may briefly become positive.  In practice $V$ always returns to decrease once the threat clears, as shown in the Lyapunov plot.
- **Obstacle avoidance is reactive, not globally optimal:** The tangent-waypoint planner may require multiple waypoints to navigate narrow corridors and can stall on very dense obstacle configurations.
- **Cannon hits:** With the current parameters the robot sustains a small number of projectile hits per run.  A more aggressive lookahead or a predictive path-planning approach would reduce this further.
- **Holonomic assumption in dodge:** The escape direction is computed assuming the robot can reach the lateral target quickly.  At low speed or with large heading error, the actual clearance can be less than predicted.
