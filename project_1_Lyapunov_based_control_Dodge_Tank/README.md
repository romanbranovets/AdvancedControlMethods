# Lyapunov Control for a Tracked Robot

---

## 4.1 Problem Definition

**Control problem:**  
Design a feedback controller that drives a tracked robot from an initial pose $(x, y, \theta)$ to a goal position $(x_g, y_g)$ in a 2D environment, with optional obstacle avoidance via intermediate waypoints.

**Plant / environment:**  
- 2D planar workspace  
- Static circular obstacles (optional)  
- Fully observable state  

**Assumptions / context:**
- Kinematic model (no slip dynamics, no inertia effects)  
- No noise or disturbances  
- Goal is fixed during execution  
- Local (reactive) decision making  

**Class of methods:**
- Nonlinear feedback control  
- Lyapunov stabilization  
- Reactive waypoint-based obstacle avoidance  

---

## 4.2 System Description

**System:**  
Tracked robot modeled as a unicycle with differential tracks.

**Visualization:**  
Implemented in `visualization.py`:
- robot body
- left/right tracks
- trajectory
- goal marker
- optional intermediate targets

---

### State variables

$$
\mathbf{x} = (x, y, \theta)
$$

- $x, y$ — planar position  
- $\theta$ — heading angle  

---

### Control inputs

Unicycle form:

$$
(v, \omega)
$$

Track velocities:

$$
(u_L, u_R)
$$

Conversion:

$$
v = \frac{u_L + u_R}{2}, \quad \omega = \frac{u_R - u_R}{b}
$$

where $b$ is track separation width.

---

### Constraints

$$
|u_L|, |u_R| \leq u_{\max}
$$

Goal condition:

$$
\sqrt{(x - x_g)^2 + (y - y_g)^2} \leq \epsilon
$$

---

### Dynamics

$$
\dot{x} = v \cos \theta
$$

$$
\dot{y} = v \sin \theta
$$

$$
\dot{\theta} = \omega
$$

---

## 4.3 Mathematical Specification

### Goal

$$
(x_g, y_g)
$$

### Distance to goal

$$
\rho = \sqrt{(x_g - x)^2 + (y_g - y)^2}
$$

### Heading error

$$
\alpha = \mathrm{wrap}\big(\arctan2(y_g - y, x_g - x) - \theta\big)
$$

---

### Control law (Lyapunov)

$$
v = k_{\rho} \cdot \rho \cdot \cos(\alpha)
$$

$$
\omega = k_{\alpha} \cdot \sin(\alpha)
$$

where:
- $k_{\rho} > 0$
- $k_{\alpha} > 0$

---

### Notation

- $\rho$ — distance to target  
- $\alpha$ — heading error  
- $v$ — linear velocity  
- $\omega$ — angular velocity  

---

## 4.4 Method Description

The method is a **Lyapunov nonlinear feedback controller** combined with **reactive waypoint selection**.

---

### Core idea

1. Compute error to target $(\rho, \alpha)$  
2. Apply nonlinear feedback law  
3. If direct path is blocked, switch to intermediate waypoint  
4. Otherwise track goal directly  

---

### Stability (practical interpretation)

- In obstacle-free case, system empirically converges to goal  
- No formal proof of global stability is implemented  
- Behavior is consistent with Lyapunov design  

---

### Obstacle avoidance

If obstacles are present:
- system selects intermediate targets stored in `sim.targets`
- mode `obstacle_avoidance` activates waypoint tracking
- robot switches back to goal when path is clear

---

### Approximations

- Pure kinematic model  
- No dynamic friction/slip modeling  
- Reactive (no global planner)

---

## 4.5 Algorithm Listing

### Pipeline

1. Initialize state $(x, y, \theta)$, goal, and obstacles.

2. For each time step:

   a. Read current state.

   b. Check for obstacle influence (if any).

   c. Select target:
      - goal, or  
      - intermediate waypoint

   d. Compute errors:
      - $\rho$ — distance to target  
      - $\alpha$ — heading error  

   e. Compute control:
      - $v = k_{\rho} \cdot \rho \cdot \cos(\alpha)$  
      - $\omega = k_{\alpha} \cdot \sin(\alpha)$  

   f. Convert $(v, \omega) \rightarrow (u_L, u_R)$.

   g. Apply saturation constraints.

   h. Update system state using kinematics.

3. Stop when:
   $$
   \rho \leq \epsilon
   $$

---

## 4.6 Experimental Setup

**Initial conditions:**
- $(x, y, \theta)$ = $(0, 0, 0)$ (default)

**Goal:**
- $(x_g, y_g)$ = $(10, 5)$

**Simulation parameters:**
- $dt = 0.05$
- steps = 1200

**Controller parameters:**
- $k_{\rho} = 1.8$
- $k_{\alpha} = 4.2$

**Environment:**
- optional circular obstacles
- optional intermediate waypoints

**Disturbances:**
- none

---

## 4.7 Reproducibility

### Installation

The project uses `pyproject.toml` and `uv.lock`:

```bash
uv sync
python visualization.py
jupyter notebook simulation.ipynb
```
The simulation is deterministic

Running the project produces:

#### Console output:
- number of simulation steps  
- final robot pose $(x, y, \theta)$  
- final distance to goal  

#### Animation:
- robot trajectory  
- robot orientation and body  
- left/right track motion  
- control inputs $(v_L, v_R)$ displayed in the title  

### 4.8 Results Summary

#### Works:
- Stable convergence to goal  
- Smooth trajectories  
- Obstacle avoidance works in most cases  

#### Limitations:
- No global planning  
- Can get stuck in complex scenes  
- Sensitive to parameters  

#### Interpretation:
- Lyapunov control guarantees stability  
- Reactive avoidance works but is not optimal  