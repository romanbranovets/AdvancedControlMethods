# Lyapunov-Based Control for a Tracked Robot

---

## 4.1 Problem Definition

**Control problem:**  
Design a feedback controller that drives a tracked robot from an initial pose $(x, y, \theta)$ to a goal position $(x_g, y_g)$ while avoiding obstacles.

**Plant / environment:**  
- 2D planar environment  
- Static circular obstacles  
- Fully observable state  

**Assumptions / context:**
- Ideal kinematics (no slip, no delays)  
- Obstacles are known and static  
- No disturbances or noise  
- Goal is fixed  

**Class of methods:**  
- Nonlinear control  
- Lyapunov-based stabilization  
- Reactive obstacle avoidance  

---

## 4.2 System Description

**System:**  
Tracked robot modeled as a unicycle.

**Visualization:**  
Animation implemented in `visualization.py` (robot body, tracks, trajectory, obstacles, waypoints).

---

### State variables

$$
\mathbf{x} = (x, y, \theta)
$$

- $x, y$ — position  
- $\theta$ — heading  

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
v = \frac{u_L + u_R}{2}, \quad \omega = \frac{u_R - u_L}{b}
$$

---

### Unknown parameters

None.

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

Define goal:
$$
(x_g, y_g)
$$

Distance to goal:
$$
\rho = \sqrt{(x_g - x)^2 + (y_g - y)^2}
$$

Heading error:
$$
\alpha = \text{wrap}(\arctan2(y_g - y, x_g - x) - \theta)
$$

---

### Lyapunov function

$$
V = \frac{1}{2} \rho^2 + c (1 - \cos \alpha)
$$

where $c > 0$.

---

### Control law

$$
v = k_\rho \cdot \rho \cdot \cos(\alpha)
$$

$$
\omega = k_\alpha \cdot \sin(\alpha)
$$

Conditions:
- $k_\rho > 0$  
- $k_\alpha > k_\rho$  

---

### Notation consistency

- $\rho$ — distance to target  
- $\alpha$ — heading error  
- $V$ — Lyapunov function  
- $v, \omega$ — control inputs

---

## 4.4 Method Description

**Method:** Lyapunov-based point stabilization with waypoint switching.

---

### Idea

1. Define Lyapunov function $V$  
2. Choose control inputs so that $\dot{V} \leq 0$  
3. Switch target when obstacles block direct path  

---

### Stability

$$
\dot{V} \leq 0
$$

Ensures convergence to the goal in obstacle-free case.

---

### Obstacle avoidance

- Detect intersection between straight path and obstacle  
- Generate candidate waypoints:
  - tangent points  
  - side bypass points  
- Select best feasible waypoint  

---

### Additional mechanisms

- Hysteresis (prevents oscillations)  
- Stall detection (switch direction if stuck)  
- Safety recovery (escape if inside obstacle)  

---

### Approximations

- Kinematic model only  
- Circular obstacles  
- Local (reactive) planning  

---

## 4.5 Algorithm Listing

### Algorithm

1. Initialize state $(x, y, \theta)$, goal, and obstacles.

2. For each time step:

   a. Read current state.

   b. Detect blocking obstacle.

   c. Select target:
      - goal, or  
      - avoidance waypoint.

   d. Compute:
      - $\rho$ — distance to target  
      - $\alpha$ — heading error  

   e. Compute control:
      - $v = k_{\rho} \cdot \rho \cdot \cos(\alpha)$  
      - $\omega = k_{\alpha} \cdot \sin(\alpha)$  

   f. Convert $(v, \omega) \rightarrow (u_L, u_R)$.

   g. Apply saturation.

   h. Update state using dynamics.

3. Stop when goal is reached.


---

## 4.6 Experimental Setup

**Initial conditions:**  
- Random or fixed $(x, y, \theta)$  

**Reference:**  
- Goal $(x_g, y_g)$  

**Simulation:**  
- $dt = 0.05$  
- steps = 2000  

**Controller parameters:**  
- $k_\rho = 0.8$  
- $k_\alpha = 4.0$  

**Environment:**  
- 2–5 obstacles  
- radius: 0.35–0.9  

**Disturbances:**  
- None  

---

## 4.7 Reproducibility

To fully reproduce the simulation results, plots, and animation, follow the steps below.

The project uses `pyproject.toml` and `uv.lock` for dependency management.

Install all dependencies with:

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