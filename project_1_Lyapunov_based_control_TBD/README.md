# 🚗 Lyapunov-Based Control for a Tracked Robot

---

## 4.1 Problem Definition

**Control problem:**  
Design a feedback controller that drives a tracked robot from an initial pose to a goal position while avoiding obstacles.

**Plant / environment:**  
- 2D plane  
- Static circular obstacles  
- Fully observable state  

**Assumptions / context:**
- Ideal kinematics (no slip, no delays)  
- Obstacles known and static  
- No noise or disturbances  
- Goal is a fixed point  

**Class of methods:**  
- Nonlinear control  
- Lyapunov-based stabilization  
- Reactive geometric obstacle avoidance  

---

## 4.2 System Description

**System:**  
Tracked robot modeled as a unicycle in 2D.

**Visualization:**  
Implemented via `visualization.py` (animation of robot, path, obstacles, targets).

---

### State variables

\[
\mathbf{x} = (x, y, \theta)
\]

- \( x, y \) — position  
- \( \theta \) — heading angle  

---

### Control inputs

Two equivalent forms:

**Unicycle:**
\[
(v, \omega)
\]

**Tracks:**
\[
(u_L, u_R)
\]

Conversion:
\[
v = \frac{u_L + u_R}{2}, \quad
\omega = \frac{u_R - u_L}{b}
\]

---

### Unknown parameters

None (fully known model).

---

### Constraints

\[
|u_L|, |u_R| \leq u_{\max}
\]

Goal condition:
\[
\sqrt{(x-x_g)^2 + (y-y_g)^2} \leq \epsilon
\]

---

### Dynamics

\[
\dot{x} = v \cos \theta
\]
\[
\dot{y} = v \sin \theta
\]
\[
\dot{\theta} = \omega
\]

---

## 4.3 Mathematical Specification

Define goal:
\[
(x_g, y_g)
\]

Distance:
\[
\rho = \sqrt{(x_g - x)^2 + (y_g - y)^2}
\]

Heading error:
\[
\alpha = \text{wrap}(\arctan2(y_g - y, x_g - x) - \theta)
\]

---

### Lyapunov function

\[
V = \frac{1}{2} \rho^2 + c (1 - \cos \alpha)
\]

where:
- \( c > 0 \) — tuning parameter  

---

### Control law

\[
v = k_\rho \cdot \rho \cdot \cos(\alpha)
\]
\[
\omega = k_\alpha \cdot \sin(\alpha)
\]

where:
- \( k_\rho > 0 \)
- \( k_\alpha > k_\rho \)

---

### Notation consistency

- \( \rho \) — distance to target  
- \( \alpha \) — heading error  
- \( V \) — Lyapunov function  
- \( v, \omega \) — control inputs  

No symbol reuse.

---

## 4.4 Method Description

**Method:** Lyapunov-based point stabilization with waypoint switching.

---

### Idea

1. Define Lyapunov function \( V \)  
2. Choose control to ensure \( \dot{V} \leq 0 \)  
3. Switch target when obstacle blocks path  

---

### Stability

\[
\dot{V} \leq 0
\]

Ensures convergence to goal (without obstacles).

---

### Obstacle avoidance

- Detect path–obstacle intersection  
- Generate candidate waypoints:
  - tangent points  
  - bypass points  
- Select best feasible waypoint  

---

### Additional mechanisms

- Hysteresis (avoid oscillation)  
- Stall detection (switch direction)  
- Safety recovery (if inside obstacle)  

---

### Approximations

- Kinematic model only  
- Circular obstacles  
- Local planning  

---

## 4.5 Algorithm Listing

**Pipeline:**
