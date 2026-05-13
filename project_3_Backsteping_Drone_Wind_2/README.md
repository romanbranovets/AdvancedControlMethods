# Backstepping Control of a Planar Drone Under Wind (2D)

This project studies **target-point regulation** of a planar drone moving in the vertical `x-z` plane. The model has **two motors**, **quadratic thrust-current relation**, **first-order motor current dynamics**, and external **wind force / pitch torque** disturbances.

The project compares four controllers under the same conditions:

- **P**
- **PD**
- **PID**
- **Backstepping with motor-current dynamics**

The main goal is to show that accounting for actuator dynamics through backstepping improves target tracking and provides a meaningful stability certificate.

---

# 1. Problem Statement

Consider a wind-free planar drone moving in the vertical plane $(x, z)$. The goal is to stabilize the drone at the desired point

$$
x \to x^{\ast}, \qquad z \to z^{\ast},
$$

with zero velocity and zero pitch angle:

$$
v_x \to 0, \qquad v_z \to 0, \qquad \theta \to 0, \qquad \omega \to 0.
$$

Unlike a simple model where the thrust or acceleration can be commanded directly, we want to design the controller **down to the motor-current level**. Therefore, the real control inputs are not accelerations or thrusts, but motor current commands.

Let

$$
I_0 = \sqrt{\frac{mg}{2k_F}}
$$

be the hover current of each motor. We introduce current deviations from hover:

$$
i_L = I_L - I_0, \qquad i_R = I_R - I_0.
$$

Instead of working directly with left and right current deviations, we use collective and differential coordinates:

$$
i_s = i_L + i_R, \qquad i_d = i_R - i_L.
$$

Here:

- $i_s$ controls the vertical thrust deviation;
- $i_d$ controls the pitch torque.

The corresponding control inputs are collective and differential current commands:

$$
u_s = u_L + u_R, \qquad u_d = u_R - u_L,
$$

where

$$
u_L = I_{L,cmd} - I_0, \qquad u_R = I_{R,cmd} - I_0.
$$

The physical motor commands can be recovered as

$$
I_{L,cmd} = I_0 + \frac{u_s - u_d}{2}, \qquad
I_{R,cmd} = I_0 + \frac{u_s + u_d}{2}.
$$

The objective is to construct a backstepping controller for $u_s$ and $u_d$ and prove global asymptotic stability of the origin of the error dynamics.

---

# 2. Transformation to State-Space Form

We use the hover-linearized planar drone model. Around hover, the thrust-current relation is linearized as

$$
T_k = k_F I_k^2 \approx k_F I_0^2 + 2k_F I_0 i_k.
$$

Since

$$
2k_F I_0^2 = mg,
$$

the hover thrust compensates gravity, and the remaining vertical acceleration is controlled by $i_s$.

The state vector is

$$
X =
\begin{bmatrix}
x & z & v_x & v_z & \theta & \omega & i_s & i_d
\end{bmatrix}^{T}.
$$

The model is

$$
\begin{aligned}
\dot x &= v_x, & \dot v_x &= g\theta, & \dot \theta &= \omega, & \dot \omega &= a_d i_d, \\
\tau_m \dot i_d &= -i_d + u_d, & \dot z &= v_z, & \dot v_z &= a_s i_s, & \tau_m \dot i_s &= -i_s + u_s.
\end{aligned}
$$

The constants are

$$
a_s = \frac{2k_F I_0}{m}, \qquad a_d = \frac{2Lk_F I_0}{J},
$$

where:

- $m$ is the drone mass;
- $J$ is the pitch moment of inertia;
- $L$ is the arm length;
- $k_F$ is the thrust coefficient;
- $\tau_m$ is the motor current time constant.

The system naturally splits into two strict-feedback chains.

Vertical chain:

$$
u_s \rightarrow i_s \rightarrow v_z \rightarrow z.
$$

Horizontal chain:

$$
u_d \rightarrow i_d \rightarrow \omega \rightarrow \theta \rightarrow v_x \rightarrow x.
$$

This structure is exactly what makes backstepping suitable.

---

# 3. Idea of Backstepping

The key idea of backstepping is to stabilize the system recursively.

For the vertical motion, we do not directly control $z$. Instead, the chain is:

$$
u_s \to i_s \to v_z \to z.
$$

So we proceed step by step:

1. Choose a desired vertical velocity to stabilize $z$.
2. Choose a desired collective current $i_s$ to stabilize $v_z$.
3. Choose the actual current command $u_s$ to make $i_s$ track its desired value.

For the horizontal motion, the chain is longer:

$$
u_d \to i_d \to \omega \to \theta \to v_x \to x.
$$

So we recursively define:

1. desired horizontal velocity;
2. desired pitch angle;
3. desired pitch rate;
4. desired differential current;
5. actual differential current command.

At every step, we introduce a new error variable and add its squared value to the Lyapunov function. The control law is chosen so that the derivative of the Lyapunov function becomes negative definite.

---

# 4. Choice of Control Law

## 4.1. Vertical channel

Define the vertical position error:

$$
e_{z1} = z - z^{\ast}.
$$

Choose the virtual desired vertical velocity:

$$
\alpha_{z1} = -k_{z1}e_{z1}.
$$

Define the vertical velocity error:

$$
e_{z2} = v_z - \alpha_{z1}.
$$

Since $\dot z = v_z$, we have

$$
\dot e_{z1} = v_z = e_{z2} + \alpha_{z1} = e_{z2} - k_{z1}e_{z1}.
$$

Now choose the desired collective current:

$$
\alpha_{z2} = \frac{\dot\alpha_{z1} - e_{z1} - k_{z2}e_{z2}}{a_s}.
$$

Define the collective current tracking error:

$$
e_{z3} = i_s - \alpha_{z2}.
$$

Since $\dot v_z = a_s i_s$, we get

$$
\dot e_{z2} = a_s i_s - \dot\alpha_{z1}.
$$

Substituting $i_s = e_{z3} + \alpha_{z2}$, we obtain

$$
\dot e_{z2} = -e_{z1} - k_{z2}e_{z2} + a_s e_{z3}.
$$

Finally, using the motor current dynamics $\tau_m\dot i_s = -i_s + u_s$, we choose the real collective current command:

$$
u_s = i_s + \tau_m\left(\dot\alpha_{z2} - a_s e_{z2} - k_{z3}e_{z3}\right).
$$

Then

$$
\dot e_{z3} = -a_s e_{z2} - k_{z3}e_{z3}.
$$

---

## 4.2. Horizontal channel

Define the horizontal position error:

$$
e_{x1} = x - x^{\ast}.
$$

Choose the virtual desired horizontal velocity:

$$
\alpha_{x1} = -k_{x1}e_{x1}.
$$

Define

$$
e_{x2} = v_x - \alpha_{x1}.
$$

Then

$$
\dot e_{x1} = e_{x2} - k_{x1}e_{x1}.
$$

Since $\dot v_x = g\theta$, choose the desired pitch angle:

$$
\alpha_{x2} = \frac{\dot\alpha_{x1} - e_{x1} - k_{x2}e_{x2}}{g}.
$$

Define the pitch angle error:

$$
e_{x3} = \theta - \alpha_{x2}.
$$

Then

$$
\dot e_{x2} = -e_{x1} - k_{x2}e_{x2} + g e_{x3}.
$$

Now choose the desired pitch rate:

$$
\alpha_{x3} = \dot\alpha_{x2} - g e_{x2} - k_{x3}e_{x3}.
$$

Define

$$
e_{x4} = \omega - \alpha_{x3}.
$$

Since $\dot\theta = \omega$, we obtain

$$
\dot e_{x3} = -g e_{x2} - k_{x3}e_{x3} + e_{x4}.
$$

Next, since $\dot\omega = a_d i_d$, choose the desired differential current:

$$
\alpha_{x4} = \frac{\dot\alpha_{x3} - e_{x3} - k_{x4}e_{x4}}{a_d}.
$$

Define the differential current tracking error:

$$
e_{x5} = i_d - \alpha_{x4}.
$$

Then

$$
\dot e_{x4} = -e_{x3} - k_{x4}e_{x4} + a_d e_{x5}.
$$

Finally, using $\tau_m\dot i_d = -i_d + u_d$, choose the actual differential current command:

$$
u_d = i_d + \tau_m\left(\dot\alpha_{x4} - a_d e_{x4} - k_{x5}e_{x5}\right).
$$

Then

$$
\dot e_{x5} = -a_d e_{x4} - k_{x5}e_{x5}.
$$

All gains are assumed positive:

$$
k_{z1}, k_{z2}, k_{z3} > 0, \qquad
k_{x1}, k_{x2}, k_{x3}, k_{x4}, k_{x5} > 0.
$$

---

# 5. Global Stability Proof

## 5.1. Vertical channel proof

Consider the vertical Lyapunov function:

$$
V_z = \frac{1}{2}e_{z1}^2 + \frac{1}{2}e_{z2}^2 + \frac{1}{2}e_{z3}^2.
$$

Its derivative is

$$
\dot V_z = e_{z1}\dot e_{z1} + e_{z2}\dot e_{z2} + e_{z3}\dot e_{z3}.
$$

Substitute the closed-loop error dynamics:

$$
\begin{aligned}
\dot e_{z1} &= e_{z2} - k_{z1}e_{z1}, \\
\dot e_{z2} &= -e_{z1} - k_{z2}e_{z2} + a_s e_{z3}, \\
\dot e_{z3} &= -a_s e_{z2} - k_{z3}e_{z3}.
\end{aligned}
$$

Then

$$
\begin{aligned}
\dot V_z
&= e_{z1}(e_{z2}-k_{z1}e_{z1})
 + e_{z2}(-e_{z1}-k_{z2}e_{z2}+a_s e_{z3})
 + e_{z3}(-a_s e_{z2}-k_{z3}e_{z3}) \\
&= -k_{z1}e_{z1}^2 - k_{z2}e_{z2}^2 - k_{z3}e_{z3}^2.
\end{aligned}
$$

Therefore, $\dot V_z < 0$ for all nonzero vertical error states. Thus, the vertical subsystem is globally asymptotically stable.

---

## 5.2. Horizontal channel proof

Consider the horizontal Lyapunov function:

$$
V_x = \frac{1}{2}e_{x1}^2 + \frac{1}{2}e_{x2}^2 + \frac{1}{2}e_{x3}^2 + \frac{1}{2}e_{x4}^2 + \frac{1}{2}e_{x5}^2.
$$

Its derivative is

$$
\dot V_x = \sum_{j=1}^{5} e_{xj}\dot e_{xj}.
$$

The closed-loop horizontal error dynamics are

$$
\begin{aligned}
\dot e_{x1} &= e_{x2} - k_{x1}e_{x1}, \\
\dot e_{x2} &= -e_{x1} - k_{x2}e_{x2} + g e_{x3}, \\
\dot e_{x3} &= -g e_{x2} - k_{x3}e_{x3} + e_{x4}, \\
\dot e_{x4} &= -e_{x3} - k_{x4}e_{x4} + a_d e_{x5}, \\
\dot e_{x5} &= -a_d e_{x4} - k_{x5}e_{x5}.
\end{aligned}
$$

Substitute them into $\dot V_x$:

$$
\begin{aligned}
\dot V_x
=& e_{x1}(e_{x2}-k_{x1}e_{x1})
 + e_{x2}(-e_{x1}-k_{x2}e_{x2}+g e_{x3}) \\
&+ e_{x3}(-g e_{x2}-k_{x3}e_{x3}+e_{x4})
 + e_{x4}(-e_{x3}-k_{x4}e_{x4}+a_d e_{x5}) \\
&+ e_{x5}(-a_d e_{x4}-k_{x5}e_{x5}) \\
=& -k_{x1}e_{x1}^2 - k_{x2}e_{x2}^2 - k_{x3}e_{x3}^2 - k_{x4}e_{x4}^2 - k_{x5}e_{x5}^2.
\end{aligned}
$$

Therefore, $\dot V_x < 0$ for all nonzero horizontal error states. Thus, the horizontal subsystem is globally asymptotically stable.

---

## 5.3. Full Lyapunov function

Now define the full Lyapunov function:

$$
V = V_x + V_z.
$$

This gives

$$
\begin{aligned}
V
=& \frac{1}{2}e_{z1}^2 + \frac{1}{2}e_{z2}^2 + \frac{1}{2}e_{z3}^2 \\
&+ \frac{1}{2}e_{x1}^2 + \frac{1}{2}e_{x2}^2 + \frac{1}{2}e_{x3}^2 + \frac{1}{2}e_{x4}^2 + \frac{1}{2}e_{x5}^2.
\end{aligned}
$$

This function is positive definite and radially unbounded. Its derivative is

$$
\dot V = \dot V_z + \dot V_x.
$$

Therefore,

$$
\begin{aligned}
\dot V
=& -k_{z1}e_{z1}^2 - k_{z2}e_{z2}^2 - k_{z3}e_{z3}^2 \\
&- k_{x1}e_{x1}^2 - k_{x2}e_{x2}^2 - k_{x3}e_{x3}^2 - k_{x4}e_{x4}^2 - k_{x5}e_{x5}^2.
\end{aligned}
$$

Since all gains are positive, $\dot V < 0$ for all nonzero error vectors. Therefore, the equilibrium

$$
e_{z1}=e_{z2}=e_{z3}=0, \qquad
e_{x1}=e_{x2}=e_{x3}=e_{x4}=e_{x5}=0
$$

is globally asymptotically stable. This implies

$$
z \to z^{\ast}, \qquad v_z \to 0, \qquad i_s \to 0,
$$

and

$$
x \to x^{\ast}, \qquad v_x \to 0, \qquad \theta \to 0, \qquad \omega \to 0, \qquad i_d \to 0.
$$

Thus, the drone reaches the desired point and stabilizes at hover.



--- OLD:

## 1. Problem Definition

### 1.1 Control objective

The drone starts from an initial point `p(0)` and must reach a fixed target point `p*`:

```text
p  = [x, z]^T
p* = [x*, z*]^T
```

where:


| Symbol | Meaning                 |
| ------ | ----------------------- |
| `x`    | horizontal position, m  |
| `z`    | vertical position, m    |
| `p`    | current position vector |
| `p*`   | desired target position |


The control objective is:

```text
lim t→∞ ||p(t) - p*|| → 0
```

in the presence of bounded disturbances:

```text
wind force:      w = [w_x(t), w_z(t)]^T
wind torque:     w_tau(t)
```

In simulation, all controllers are compared on a fixed time horizon, so the final error is measured at `T_max = 18 s`.

### 1.2 Performance metrics

The project evaluates:


| Metric               | Meaning                                                                |
| -------------------- | ---------------------------------------------------------------------- |
| `                    |                                                                        |
| `IAE`                | integral absolute position error                                       |
| `overshoot`          | how far the drone passes beyond the target along the start-target line |
| motor current        | how aggressively the controller uses actuators                         |
| Lyapunov certificate | diagnostic stability evidence for backstepping                         |


Integral absolute error:

```text
IAE = ∫[0,T] ||p(t) - p*|| dt
```

---

## 2. Visual Results

All images and animations below are embedded directly in the README. They are generated by:

```bash
uv run python main.py
```

### 2.1 Controller comparison



**Figure 1 — Controller comparison.**

This figure compares **P**, **PD**, **PID**, and **Backstepping**:

- **Top-left:** distance to target `||p - p*||` over time.
- **Top-right:** mean motor current `(I_L + I_R) / 2`, showing actuator effort.
- **Bottom-left:** planar trajectories in the `x-z` plane; the gold star is the target.
- **Bottom-right:** numeric performance table with IAE, final error, and simulation horizon.

The key observation is that **Backstepping reaches the target with the smallest final position error** because it explicitly compensates motor-current dynamics.

### 2.2 Overshoot comparison



**Figure 2 — Overshoot comparison.**

Overshoot is measured along the straight line from the initial point to the target. If the projection of the trajectory passes beyond the target, the excess distance is counted as overshoot.

- **Left:** overshoot time series for each controller.
- **Right:** peak overshoot, final distance to target, and minimum achieved distance.

This plot shows whether a controller approaches the target smoothly or passes through it with excessive kinetic energy.

### 2.3 Detailed backstepping run



**Figure 3 — Backstepping diagnostic plots.**

Panels show:

- trajectory in the `x-z` plane;
- velocity components `v_x`, `v_z` and their time-varying references;
- motor current commands `I_L`, `I_R`;
- pitch angle `theta`;
- pitch torque `tau`;
- position error `||p - p*||`.

This figure is useful for checking that convergence is achieved without unreasonable current or pitch behavior.

### 2.4 Lyapunov certificate



**Figure 4 — Lyapunov certificate for Backstepping.**

The figure shows:

- **Top:** total Lyapunov-like function `V(t)`.
- **Middle:** decomposition into velocity, pitch, angular-rate, and current terms.
- **Bottom:** numerical derivative `dV/dt`.

The derivative is computed numerically, so it should be interpreted as a diagnostic signal rather than a fully symbolic proof. Negative regions support the stability intuition of the backstepping design.

### 2.5 Animation



**Figure 5 — Backstepping flight animation.**

The left panel shows the drone as a line segment whose orientation is the pitch angle `theta`. The trajectory is drawn in the `x-z` plane. The right panel shows the position error over time.

### 2.6 Interactive dashboard

The file `dashboard_2d.html` contains a Plotly dashboard comparing **PD** and **Backstepping** trajectories with playback controls. Open it in a browser after running the script.

---

## 3. System Model

### 3.1 State vector

The planar drone state is:

```text
state = [x, z, v_x, v_z, theta, omega, I_L, I_R]^T
```

where:


| Variable | Unit  | Meaning             |
| -------- | ----- | ------------------- |
| `x`      | m     | horizontal position |
| `z`      | m     | vertical position   |
| `v_x`    | m/s   | horizontal velocity |
| `v_z`    | m/s   | vertical velocity   |
| `theta`  | rad   | pitch angle         |
| `omega`  | rad/s | pitch rate          |
| `I_L`    | A     | left motor current  |
| `I_R`    | A     | right motor current |


### 3.2 Motor thrust and torque

Each motor produces thrust proportional to squared current:

```text
T_L = k_F * I_L^2
T_R = k_F * I_R^2
T   = T_L + T_R
```

Pitch torque is produced by differential thrust:

```text
tau = L * k_F * (I_R^2 - I_L^2)
```

where:


| Symbol       | Meaning                     |
| ------------ | --------------------------- |
| `T_L`, `T_R` | left and right motor thrust |
| `T`          | total thrust                |
| `tau`        | pitch torque                |
| `L`          | effective motor arm         |
| `k_F`        | thrust coefficient          |


### 3.3 Translational and rotational dynamics

The dynamics in the inertial `x-z` frame are:

```text
m * d(v_x)/dt = T * sin(theta) + w_x - c_d * v_x

m * d(v_z)/dt = T * cos(theta) - m*g + w_z - c_d * v_z

J * d(omega)/dt = tau + w_tau

d(theta)/dt = omega

d(x)/dt = v_x
d(z)/dt = v_z
```

where:


| Symbol       | Meaning                            |
| ------------ | ---------------------------------- |
| `m`          | drone mass                         |
| `g`          | gravitational acceleration         |
| `J`          | pitch moment of inertia            |
| `c_d`        | linear drag coefficient            |
| `w_x`, `w_z` | wind force components              |
| `w_tau`      | external wind-induced pitch torque |


### 3.4 Motor-current dynamics

The motor current follows a first-order actuator model:

```text
d(I_k)/dt = (I_k_cmd - I_k) / tau_m,       k ∈ {L, R}
```

where:


| Symbol    | Meaning                     |
| --------- | --------------------------- |
| `I_k`     | actual motor current        |
| `I_k_cmd` | commanded motor current     |
| `tau_m`   | motor-current time constant |


This actuator lag is the main reason why backstepping is useful: the controller explicitly compensates it.

### 3.5 Wind model

The default wind is:

```text
w_x(t)   = 0.12*sin(0.85*t) + 0.05
w_z(t)   = 0.08*cos(1.10*t)
w_tau(t) = 0.008*sin(2.20*t)
```

---

## 4. Control Architecture

The control system is cascaded:

```text
target point p*
      │
      ▼
outer position loop
      │      p* - p
      ▼
reference velocity v_ref
      │
      ▼
P / PD / PID / Backstepping velocity controller
      │
      ▼
desired acceleration a_des
      │
      ▼
thrust-pitch allocation
      │
      ▼
desired total thrust T_des and pitch theta_des
      │
      ▼
motor inversion
      │
      ▼
current commands [I_L_cmd, I_R_cmd]
      │
      ▼
planar drone plant
```

### 4.1 Outer position loop

The outer loop converts position error into a bounded velocity reference:

```text
e_p = p* - p
v_raw = k_p * e_p

if ||v_raw|| <= v_max:
    v_ref = v_raw
else:
    v_ref = v_max * v_raw / ||v_raw||
```

where:


| Symbol  | Meaning                                     |
| ------- | ------------------------------------------- |
| `e_p`   | position error                              |
| `k_p`   | outer position gain                         |
| `v_max` | maximum allowed reference speed             |
| `v_ref` | desired velocity passed to inner controller |


---

## 5. Allocation and Motor Inversion

The velocity controller produces a desired acceleration:

```text
a_des = [a_x_des, a_z_des]^T
```

The required force components are:

```text
f_x = m * a_x_des
f_z = m * (a_z_des + g)
```

Total desired thrust and desired pitch:

```text
T_des     = sqrt(f_x^2 + f_z^2)
theta_des = atan2(f_x, f_z)
```

The desired pitch is clipped:

```text
|theta_des| <= theta_lim
```

The motor inversion solves:

```text
T_des   = k_F * (s_L + s_R)
tau_des = L * k_F * (s_R - s_L)

s_L = I_L_des^2
s_R = I_R_des^2
```

Therefore:

```text
s_R = 0.5 * (T_des/k_F + tau_des/(L*k_F))
s_L = 0.5 * (T_des/k_F - tau_des/(L*k_F))

I_R_des = sqrt(s_R)
I_L_des = sqrt(s_L)
```

Then both desired currents are clipped to actuator bounds.

---

## 6. Compared Controllers

Let:

```text
e_v = v_ref - v
```

### 6.1 P velocity controller

```text
a_des = k_p_v * e_v
```

### 6.2 PD velocity controller

For a piecewise constant velocity reference, `d(e_v)/dt ≈ -a`. The implemented law is:

```text
a_des = k_p_v * e_v - k_d_v * a
```

### 6.3 PID velocity controller

```text
a_des = k_p_v * e_v - k_d_v * a + k_i_v * integral(e_v dt)
```

The integral term is clamped to avoid windup.

### 6.4 Backstepping controller

The translational part starts with:

```text
a_des = k_v * e_v
```

Then pitch tracking is shaped by a virtual pitch-rate command:

```text
e_theta   = theta_des - theta_hat
omega_des = k_theta * e_theta + d(theta_des)/dt
```

Torque command:

```text
tau_des = J * d(omega_des)/dt + J*k_omega*(omega_des - omega_hat)
```

The implementation blends this with measured motor torque:

```text
tau_des <- (1 - beta)*tau_des + beta*tau
```

Finally, the current dynamics are backstepped:

```text
e_I_k = I_k - I_k_des

I_k_cmd = I_k + tau_m * ( d(I_k_des)/dt - k_I * e_I_k )

k ∈ {L, R}
```

This last equation explicitly compensates the first-order motor-current lag.

---

## 7. Stability Proof and Certificate

### 7.1 Ideal motor-current backstepping proof

Consider one current channel:

```text
dI/dt = (u - I) / tau_m
```

Let:

```text
e_I = I - I_des
```

Choose:

```text
u = I + tau_m * ( d(I_des)/dt - k_I * e_I )
```

Then:

```text
de_I/dt
  = dI/dt - d(I_des)/dt
  = (u - I)/tau_m - d(I_des)/dt
  = -k_I * e_I
```

Use the Lyapunov function:

```text
V_I = 0.5 * e_I^2
```

Its derivative:

```text
dV_I/dt
  = e_I * de_I/dt
  = -k_I * e_I^2
  <= 0
```

Thus the current tracking error converges exponentially to zero in the ideal scalar actuator model. The two-motor case is the same argument applied independently to the left and right motor currents.

### 7.2 Composite Lyapunov-like certificate used in simulation

The simulation logs a positive aggregate:

```text
e_v     = v_ref - v
e_theta = theta_des - theta_hat
e_omega = omega_des - omega_hat
e_I     = I - I_des
```

Weights:

```text
c_theta = 0.08 * m * g
c_I     = k_I * tau_m
```

Composite function:

```text
V =
  0.5 * m       * ||e_v||^2
+ 0.5 * c_theta * e_theta^2
+ 0.5 * J       * e_omega^2
+ 0.5 * c_I     * ||e_I||^2
```

This is not claimed to be a complete global Lyapunov function for the entire nonlinear saturated cascade. It is a **diagnostic certificate**: it combines the errors targeted by the backstepping construction and is plotted together with a numerical derivative.

### 7.3 Overshoot definition

Let:

```text
p_0 = p(0)
L   = ||p* - p_0||
u_p = (p* - p_0) / L
```

Scalar progress along the start-target line:

```text
s(t) = (p(t) - p_0)^T * u_p
```

Overshoot:

```text
O(t) = max(0, s(t) - L)
```

If `O(t) > 0`, the drone has passed beyond the target along the initial line of sight.

---

## 8. Numerical Experiment


| Quantity            | Value                                                 |
| ------------------- | ----------------------------------------------------- |
| Target point        | `(6.0, 4.0)` m                                        |
| Outer position gain | `pos_gain = 0.7`                                      |
| Velocity cap        | `v_max = 1.35` m/s                                    |
| Horizon             | `18 s`                                                |
| RK4 step            | `0.004 s`                                             |
| Initial position    | random: `x0 in [-0.5, 0.5]`, `z0 in [2, 5]`, seed `7` |
| early_stop          | `False` for fair fixed-horizon comparison             |
| Mass                | `m = 0.5 kg`                                          |
| Gravity             | `g = 9.81 m/s²`                                       |
| Pitch inertia       | `J = 0.014 kg m²`                                     |
| Arm length          | `L = 0.18 m`                                          |
| Thrust coefficient  | `k_F = 0.38 N/A²`                                     |
| Motor time constant | `tau_m = 0.07 s`                                      |
| Max current         | `I_max = 16 A`                                        |


---

## 9. Reproducibility

Install and run:

```bash
uv sync
uv run python main.py
```

Generated files:


| File                          | Description                   |
| ----------------------------- | ----------------------------- |
| `controllers_comparison.png`  | controller comparison         |
| `overshoot_comparison.png`    | overshoot comparison          |
| `results_2d_backstepping.png` | detailed backstepping signals |
| `lyapunov_certificate.png`    | Lyapunov-style certificate    |
| `drone_2d.gif`                | animation                     |
| `dashboard_2d.html`           | interactive Plotly dashboard  |


---

## 10. Repository Layout

```text
project_3_Backsteping_Drone_Wind_2/
├── README.md
├── main.py
├── pyproject.toml
├── uv.lock
├── src/
│   ├── system.py
│   ├── controller.py
│   ├── simulation.py
│   ├── plots.py
│   ├── visualization.py
│   └── plotly_dashboard.py
├── controllers_comparison.png
├── overshoot_comparison.png
├── results_2d_backstepping.png
├── lyapunov_certificate.png
├── drone_2d.gif
└── dashboard_2d.html
```

