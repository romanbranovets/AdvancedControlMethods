"""
system.py — Kinematic model of the tracked (differential-drive) robot.

This module is the *plant description* — it contains only the mathematical
model of the system.  The controller lives in controller.py; the simulation
runner in simulation.py.

===========================================================================
State:   s = [x, y, θ]ᵀ ∈ ℝ² × (−π, π]
  x   — horizontal position of the robot centre of mass, m
  y   — vertical   position of the robot centre of mass, m
  θ   — heading angle (rad), counter-clockwise from the positive x-axis

Control input:   u = [u_L, u_R]ᵀ ∈ ℝ²
  u_L — left  track velocity, m/s
  u_R — right track velocity, m/s

Derived unicycle signals:
  v = ½ (u_L + u_R)          linear  velocity, m/s
  ω = (u_R − u_L) / b        angular velocity, rad/s
  where b > 0 is the distance between track centres, m

Continuous-time kinematics (unicycle approximation):
  ẋ = v · cos θ
  ẏ = v · sin θ
  θ̇ = ω

Discrete-time approximation (forward Euler, step dt > 0):
  x_{k+1} = x_k + dt · v_k · cos θ_k
  y_{k+1} = y_k + dt · v_k · sin θ_k
  θ_{k+1} = wrap(θ_k + dt · ω_k)        wrap : ℝ → (−π, π]

Measurement model — additive Gaussian noise:
  x̃_k = x_k + η_x,    η_x ~ N(0, σ_pos²)
  ỹ_k = y_k + η_y,    η_y ~ N(0, σ_pos²)
  θ̃_k = wrap(θ_k + η_θ),  η_θ ~ N(0, σ_hdg²)

The noise models sensor imperfections (GPS, IMU drift) and is implemented
via numpy.random.Generator for full reproducibility.
===========================================================================
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Angle utility
# ---------------------------------------------------------------------------

def wrap_angle(angle: float) -> float:
    """Wrap *angle* to (−π, π]."""
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


# ---------------------------------------------------------------------------
# Track ↔ unicycle conversions
# ---------------------------------------------------------------------------

def tracks_to_unicycle(u_l: float, u_r: float, b: float) -> tuple[float, float]:
    """Map track velocities (u_L, u_R) to unicycle command (v, ω).

    Parameters
    ----------
    u_l : left  track velocity, m/s
    u_r : right track velocity, m/s
    b   : distance between track centres, m  (must be > 0)

    Returns
    -------
    v     : linear  velocity, m/s
    omega : angular velocity, rad/s
    """
    if b <= 0.0:
        raise ValueError("b must be positive")
    return float(0.5 * (u_l + u_r)), float((u_r - u_l) / b)


def unicycle_to_tracks(v: float, omega: float, b: float) -> tuple[float, float]:
    """Map unicycle command (v, ω) to track velocities (u_L, u_R).

    Parameters
    ----------
    v     : linear  velocity, m/s
    omega : angular velocity, rad/s
    b     : distance between track centres, m  (must be > 0)

    Returns
    -------
    u_l : left  track velocity, m/s
    u_r : right track velocity, m/s
    """
    if b <= 0.0:
        raise ValueError("b must be positive")
    return float(v - 0.5 * b * omega), float(v + 0.5 * b * omega)


# ---------------------------------------------------------------------------
# Discrete kinematic step
# ---------------------------------------------------------------------------

def kinematic_step(
    state: np.ndarray,
    u_l: float,
    u_r: float,
    b: float,
    dt: float,
) -> np.ndarray:
    """Advance the robot state by one forward-Euler step.

    Parameters
    ----------
    state : current state [x, y, θ]
    u_l   : left  track velocity, m/s
    u_r   : right track velocity, m/s
    b     : distance between track centres, m
    dt    : time step, s

    Returns
    -------
    next_state : [x_{k+1}, y_{k+1}, θ_{k+1}]
    """
    x, y, theta = np.asarray(state, dtype=float).reshape(3)
    v, omega = tracks_to_unicycle(u_l, u_r, b)
    x_new = x + dt * v * np.cos(theta)
    y_new = y + dt * v * np.sin(theta)
    theta_new = wrap_angle(theta + dt * omega)
    return np.array([x_new, y_new, theta_new], dtype=float)


# ---------------------------------------------------------------------------
# Measurement noise
# ---------------------------------------------------------------------------

def add_measurement_noise(
    state: np.ndarray,
    position_std: float,
    heading_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a noisy observation of the true robot state.

    Noise model:
      x̃ = x + N(0, σ_pos²)
      ỹ = y + N(0, σ_pos²)
      θ̃ = wrap(θ + N(0, σ_hdg²))

    Parameters
    ----------
    state        : true state [x, y, θ]
    position_std : standard deviation of position noise, m  (σ_pos ≥ 0)
    heading_std  : standard deviation of heading  noise, rad (σ_hdg ≥ 0)
    rng          : numpy random Generator — use default_rng(seed) for reproducibility

    Returns
    -------
    noisy_state : [x̃, ỹ, θ̃]
    """
    s = np.asarray(state, dtype=float).copy()
    noise = rng.normal(0.0, [position_std, position_std, heading_std])
    s += noise
    s[2] = wrap_angle(s[2])
    return s
