"""
scripts/generate_report.py
==========================
Generate all static figures for the MRAC drone+wind project report.

Produces (in figures/):
    01_trajectory_3d.png        3D trajectory comparison PID vs MRAC
    02_xy_topdown.png           top-down 2D trajectory (x vs y)
    03_state_signals.png        pos(t), vel(t), euler(t)
    04_lyapunov.png             V(t) = 1/2 ||e||^2 + 1/(2g) ||Theta_hat||^2
    05_error_metrics.png        rho(t) tracking error PID vs MRAC
    06_phase_portrait.png       r vs r_dot phase plot (distance to target)
    07_wind_estimation.png      Estimated disturbance vs true wind drag
    08_adaptation.png           Theta_hat(t) per axis + ref-model tracking
    09_control_signals.png      motor thrusts u_i(t)

Usage:
    python scripts/generate_report.py            # default seed=42
    python scripts/generate_report.py --seed 7   # different scenario
"""

from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.system import QuadcopterSystem
from src.controller import Controller, MRACController
from src.simulation import run_simulation


_FIG_DIR = os.path.join(_PROJECT_ROOT, "figures")
os.makedirs(_FIG_DIR, exist_ok=True)


plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

PID_COLOR  = "#bdbdbd"   # gray, baseline
MRAC_COLOR = "#1565c0"   # blue, our adaptive
TARGET_COLOR = "#d32f2f"
START_COLOR  = "#2e7d32"
WIND_COLOR   = "#f57c00"


def _save(fig, name):
    p = os.path.join(_FIG_DIR, name)
    fig.savefig(p)
    plt.close(fig)
    print(f"  saved {p}")


def _kw_from_system(s):
    return dict(m=s.m, g=s.g, l=s.l, d=s.d,
                Ixx=s.I[0, 0], Iyy=s.I[1, 1], Izz=s.I[2, 2])


def _make_wind():
    """Same hard wind profile used in main.py."""
    def w(t):
        return np.array([
            2.5 * np.sin(2.5 * t) + 1.5 * np.sin(0.4 * t) + 1.5,
            2.0 * np.cos(2.0 * t) + 1.5 * np.cos(0.3 * t) + 1.0,
            0.9 * np.sin(3.0 * t) + 0.6 * np.sin(0.5 * t) + 0.4,
        ])
    return w


# ============================================================================
# Figure 01: 3D trajectory comparison
# ============================================================================
def fig_01_trajectory_3d(data_pid, data_mrac, target):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    p = data_pid["states"][:, 0:3]
    m = data_mrac["states"][:, 0:3]

    ax.plot(p[:, 0], p[:, 1], p[:, 2], color=PID_COLOR,
            linewidth=2.0, alpha=0.85, label="PID baseline")
    ax.plot(m[:, 0], m[:, 1], m[:, 2], color=MRAC_COLOR,
            linewidth=2.5, label="MRAC + sigma-mod")

    ax.scatter(*p[0], color=START_COLOR, s=120, marker="o", label="Start")
    ax.scatter(*target, color=TARGET_COLOR, s=220, marker="*",
               label=f"Target ({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f})")

    # ε-ball wireframe
    eps = 0.15
    u, v = np.mgrid[0:2*np.pi:25j, 0:np.pi:13j]
    xs = target[0] + eps * np.cos(u) * np.sin(v)
    ys = target[1] + eps * np.sin(u) * np.sin(v)
    zs = target[2] + eps * np.cos(v)
    ax.plot_wireframe(xs, ys, zs, color=TARGET_COLOR, alpha=0.20, linewidth=0.5)

    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
    ax.set_title("Figure 1 — 3D Trajectory: PID vs MRAC under wind\n"
                 "PID drifts; MRAC tightens after adaptation kicks in.")
    ax.legend(loc="upper left", fontsize=9)
    _save(fig, "01_trajectory_3d.png")


# ============================================================================
# Figure 02: top-down x-y trajectory + cylindrical ρ-z
# ============================================================================
def fig_02_xy_topdown(data_pid, data_mrac, target):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    p = data_pid["states"][:, 0:3]
    m = data_mrac["states"][:, 0:3]

    ax = axes[0]
    ax.plot(p[:, 0], p[:, 1], color=PID_COLOR,    linewidth=2, alpha=0.85, label="PID")
    ax.plot(m[:, 0], m[:, 1], color=MRAC_COLOR,   linewidth=2.5, label="MRAC")
    ax.scatter(p[0, 0], p[0, 1], color=START_COLOR, s=120, marker="o", zorder=5, label="Start")
    ax.scatter(target[0], target[1], color=TARGET_COLOR, s=200, marker="*", zorder=5, label="Target")
    ax.add_artist(plt.Circle((target[0], target[1]), 0.15, color=TARGET_COLOR, alpha=0.15))
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.set_title("Top-down view (X vs Y)\nshaded disk = epsilon-ball (0.15 m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=9)

    ax = axes[1]
    rho_p = np.hypot(p[:, 0] - target[0], p[:, 1] - target[1])
    rho_m = np.hypot(m[:, 0] - target[0], m[:, 1] - target[1])
    ax.plot(rho_p, p[:, 2], color=PID_COLOR,  linewidth=2, alpha=0.85, label="PID")
    ax.plot(rho_m, m[:, 2], color=MRAC_COLOR, linewidth=2.5, label="MRAC")
    ax.scatter(0, target[2], color=TARGET_COLOR, s=200, marker="*", zorder=5, label="Target")
    ax.scatter(np.hypot(p[0, 0] - target[0], p[0, 1] - target[1]), p[0, 2],
               color=START_COLOR, s=120, marker="o", zorder=5, label="Start")
    ax.set_xlabel(r"$\sqrt{(x-x_t)^2 + (y-y_t)^2}$  [m]   (horizontal distance to target)")
    ax.set_ylabel("Z [m]")
    ax.set_title("Cylindrical projection (rho_xy vs Z)\nperfect tracking is a curve into the target marker")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=9)

    fig.suptitle("Figure 2 — 2-D projections of the flight",
                 y=1.02, fontsize=13)
    plt.tight_layout()
    _save(fig, "02_xy_topdown.png")


# ============================================================================
# Figure 03: state signals (pos, vel, euler)
# ============================================================================
def fig_03_state_signals(data_pid, data_mrac, target):
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    labels = [("x", "y", "z"), ("vx", "vy", "vz"),
              (r"$\phi$", r"$\theta$", r"$\psi$")]

    for row in range(3):
        ax = axes[row]
        for j, name in enumerate(labels[row]):
            sig_p = data_pid["states"][:, 3 * row + j]
            sig_m = data_mrac["states"][:, 3 * row + j]
            if row == 2:  # euler in degrees
                sig_p = np.rad2deg(sig_p); sig_m = np.rad2deg(sig_m)
            ax.plot(data_pid["t"],  sig_p, color=f"C{j}", alpha=0.45,
                    linestyle="--", linewidth=1.5)
            ax.plot(data_mrac["t"], sig_m, color=f"C{j}", linewidth=2.0,
                    label=name)
            if row == 0:  # add target reference for position
                ax.axhline(target[j], color=f"C{j}", linestyle=":", linewidth=1.0, alpha=0.5)

        if row == 0:
            ax.set_ylabel("Position [m]"); ax.set_title("Position (dashed = PID, solid = MRAC; dotted = target)")
        elif row == 1:
            ax.set_ylabel("Velocity [m/s]"); ax.set_title("Velocity")
        else:
            ax.set_ylabel("Euler [deg]"); ax.set_title("Attitude")

        ax.legend(loc="best", fontsize=9, ncol=3)
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Figure 3 — State trajectories: PID (translucent) vs MRAC (solid)",
                 fontsize=13, y=1.00)
    plt.tight_layout()
    _save(fig, "03_state_signals.png")


# ============================================================================
# Figure 04: Lyapunov function V(t)
# ============================================================================
def fig_04_lyapunov(ctrl_mrac, gamma):
    h = ctrl_mrac.history_arrays()
    t = h["t"]
    v_plant = h["v"]; v_ref = h["v_m"]
    e = v_plant - v_ref                      # tracking error per axis (T, 3)
    theta = h["theta"]                       # (T, 3, n_basis)

    V_e_per_axis = 0.5 * e ** 2                                 # (T, 3)
    V_theta_per_axis = 0.5 / gamma * np.sum(theta ** 2, axis=2) # (T, 3)
    V_per_axis = V_e_per_axis + V_theta_per_axis                # (T, 3)
    V_total = np.sum(V_per_axis, axis=1)                        # (T,)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax = axes[0]
    ax.plot(t, V_total, color="black", linewidth=2.2,
            label=r"$V(t) = \frac{1}{2}\|e\|^2 + \frac{1}{2\gamma}\|\hat\Theta\|^2$ (sum over axes)")
    ax.fill_between(t, 0, V_total, color="black", alpha=0.06)
    for j, name in enumerate(["x", "y", "z"]):
        ax.plot(t, V_per_axis[:, j], color=f"C{j}", alpha=0.7,
                linewidth=1.4, label=f"V_{name}")
    ax.set_ylabel(r"$V$")
    ax.set_title(r"Lyapunov function $V(t)$ — should be bounded (UUB) and "
                 r"trend downward in steady state")
    ax.legend(loc="best", fontsize=9, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.5)

    ax = axes[1]
    ax.plot(t, np.linalg.norm(e, axis=1), color="C3", linewidth=2.0,
            label=r"$\|e(t)\|$ — tracking error")
    ax.plot(t, np.sqrt(2.0 * np.sum(V_theta_per_axis, axis=1) * gamma),
            color="C0", linewidth=2.0, alpha=0.8,
            label=r"$\|\hat\Theta\|$ — adaptive parameters")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("magnitude")
    ax.set_title("Decomposition: tracking error ||e|| and parameter magnitude ||Theta_hat||")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(
        "Figure 4 — Lyapunov candidate "
        r"$V = \frac{1}{2}\|v - v_m\|^2 + \frac{1}{2\gamma}\|\hat\Theta\|^2$",
        fontsize=13, y=1.00,
    )
    fig.text(
        0.5, -0.03,
        "Note: theory guarantees uniform ultimate boundedness (UUB), not V(t) -> 0. "
        "Spikes correspond to acceleration/maneuver phases when ||e|| transiently grows; "
        "sigma-modification keeps Theta_hat bounded so V cannot diverge.",
        ha="center", fontsize=10, wrap=True,
    )
    plt.tight_layout()
    _save(fig, "04_lyapunov.png")


# ============================================================================
# Figure 05: tracking error rho(t)
# ============================================================================
def fig_05_error_metrics(data_pid, data_mrac, target, eps):
    rho_p = np.linalg.norm(data_pid["states"][:, 0:3]  - target, axis=1)
    rho_m = np.linalg.norm(data_mrac["states"][:, 0:3] - target, axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)

    ax = axes[0]
    ax.plot(data_pid["t"],  rho_p, color=PID_COLOR,  linewidth=2.0, alpha=0.95, label="PID baseline")
    ax.plot(data_mrac["t"], rho_m, color=MRAC_COLOR, linewidth=2.4, label="MRAC")
    ax.axhline(eps, color=TARGET_COLOR, linestyle="--", linewidth=1.2,
               label=fr"$\varepsilon$-ball = {eps:.2f} m")
    ax.set_ylabel(r"$\rho(t) = \|p - p_t\|$  [m]")
    ax.set_xlabel("Time [s]")
    ax.set_title(r"Tracking error $\rho(t)$ — distance to target")
    ax.set_yscale("symlog", linthresh=0.1)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5, which="both")

    # cumulative IAE
    iae_p = np.array([np.trapezoid(rho_p[:i+1], data_pid["t"][:i+1])  for i in range(len(rho_p))])
    iae_m = np.array([np.trapezoid(rho_m[:i+1], data_mrac["t"][:i+1]) for i in range(len(rho_m))])
    ax = axes[1]
    ax.plot(data_pid["t"],  iae_p, color=PID_COLOR,  linewidth=2.0, alpha=0.95, label="PID  IAE")
    ax.plot(data_mrac["t"], iae_m, color=MRAC_COLOR, linewidth=2.4, label="MRAC IAE")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$\int_0^t \rho(\tau)\,d\tau$  [m·s]")
    ax.set_title("Cumulative integral absolute error (IAE) — lower is better")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("Figure 5 — Goal-tracking error PID vs MRAC", fontsize=13, y=1.00)
    fig.text(
        0.5, -0.02,
        f"Final: PID = {rho_p[-1]:.3f} m, MRAC = {rho_m[-1]:.3f} m   |   "
        f"IAE: PID = {iae_p[-1]:.2f} m·s, MRAC = {iae_m[-1]:.2f} m·s",
        ha="center", fontsize=10, weight="bold",
    )
    plt.tight_layout()
    _save(fig, "05_error_metrics.png")


# ============================================================================
# Figure 06: r vs r_dot phase portrait
# ============================================================================
def fig_06_phase_portrait(data_pid, data_mrac, target):
    fig, ax = plt.subplots(figsize=(9, 7))

    for data, color, lw, alpha, label in [
        (data_pid,  PID_COLOR,  2.0, 0.85, "PID"),
        (data_mrac, MRAC_COLOR, 2.4, 1.0,  "MRAC"),
    ]:
        pos = data["states"][:, 0:3]
        vel = data["states"][:, 3:6]
        delta = pos - target
        r = np.linalg.norm(delta, axis=1)
        # r_dot via dot product (chain rule): r·r_dot = (p-t)·v  =>  r_dot = (p-t)·v / r
        r_safe = np.where(r > 1e-6, r, 1e-6)
        r_dot = np.einsum("ij,ij->i", delta, vel) / r_safe

        # color along time
        cmap = plt.cm.gray_r if label == "PID" else plt.cm.Blues
        for k in range(len(r) - 1):
            ax.plot(r[k:k+2], r_dot[k:k+2],
                    color=cmap(0.3 + 0.65 * k / max(1, len(r) - 1)),
                    linewidth=lw, alpha=alpha)
        ax.plot([], [], color=color, linewidth=lw, label=f"{label} (early -> late)")

        ax.scatter(r[0], r_dot[0], color=START_COLOR, s=80, marker="o", zorder=5)
        ax.scatter(r[-1], r_dot[-1], color=color, s=80, marker="X", zorder=5,
                   edgecolor="black")

    ax.axhline(0, color="black", linewidth=0.7, alpha=0.6)
    ax.axvline(0, color="black", linewidth=0.7, alpha=0.6)
    ax.set_xlabel(r"$r = \|p - p_t\|$  [m]   (distance to target)")
    ax.set_ylabel(r"$\dot r$  [m/s]   (rate of approach; negative -> approaching)")
    ax.set_title("Figure 6 — Phase portrait of distance to target\n"
                 "Top-left -> Bottom-left = ideal: large r, fast approach -> r=0\n"
                 "Loops = orbiting around target without committing")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    _save(fig, "06_phase_portrait.png")


# ============================================================================
# Figure 07: wind / disturbance estimation
# ============================================================================
def fig_07_wind_estimation(ctrl_mrac, system, sim_data, wind_func):
    """
    Compare adaptive estimate Theta_hat^T Phi(v) against the true matched
    disturbance (drag force on velocity). Theory does NOT guarantee they match
    pointwise (no PE) — only that the closed-loop is UUB.
    """
    h = ctrl_mrac.history_arrays()
    t = h["t"]; theta = h["theta"]; v = h["v"]

    # adaptive estimate per axis
    n_basis = theta.shape[2]
    if n_basis == 1:
        Phi = np.ones((len(t), 3, 1))
    elif n_basis == 2:
        Phi = np.stack([np.ones_like(v), v], axis=-1)
    else:
        Phi = np.stack([np.ones_like(v), v, np.abs(v) * v], axis=-1)
    delta_hat = np.einsum("tij,tij->ti", theta, Phi)  # (T, 3) — Theta_hat^T Phi

    # true matched disturbance: actual_accel - commanded_accel
    # easier: compute drag/m from physical model (using stored sim states)
    # We need v(t) from sim_data and v_wind(t). Use sim_data times to interpolate.
    sim_t = sim_data["t"]
    sim_v = sim_data["states"][:, 3:6]
    delta_true = np.zeros((len(t), 3))
    for k, tk in enumerate(t):
        # nearest sim index
        idx = int(np.clip(round(tk / max(sim_t[1] - sim_t[0], 1e-9)), 0, len(sim_t) - 1))
        v_drone = sim_v[idx]
        v_wind  = wind_func(tk)
        v_rel = v_drone - v_wind
        F_drag = -system.c_drag_lin * v_rel - system.c_drag_quad * np.linalg.norm(v_rel) * v_rel
        delta_true[k] = F_drag / system.m

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axis_names = ["x", "y", "z"]
    for j in range(3):
        ax = axes[j]
        ax.plot(t, delta_true[:, j], color=WIND_COLOR, linewidth=1.4, alpha=0.7,
                label=r"true disturbance $\delta_{\rm true}(t)$ = drag/$m$")
        ax.plot(t, delta_hat[:, j], color=MRAC_COLOR, linewidth=2.0,
                label=r"MRAC estimate $\hat\Theta^\top \Phi(v)$")
        ax.set_ylabel(f"axis {axis_names[j]}\n[m/s²]")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Figure 7 — Wind / drag disturbance: true vs MRAC estimate",
                 fontsize=13, y=1.00)
    fig.text(
        0.5, -0.04,
        "MRAC adapts Theta_hat to compensate disturbance, not to identify it.\n"
        "Without persistent excitation, Theta_hat does NOT have to converge to "
        "the true Theta*; only the closed-loop tracking error is guaranteed bounded.\n"
        "The orange curve uses unobservable v_wind(t); MRAC only ever sees v(t) "
        "and infers a state-dependent compensation that is sufficient for tracking.",
        ha="center", fontsize=10, wrap=True,
    )
    plt.tight_layout()
    _save(fig, "07_wind_estimation.png")


# ============================================================================
# Figure 08: adaptation history Θ̂(t) per axis
# ============================================================================
def fig_08_adaptation(ctrl_mrac):
    h = ctrl_mrac.history_arrays()
    t = h["t"]
    v, v_m, v_des = h["v"], h["v_m"], h["v_des"]
    theta = h["theta"]
    n_basis = theta.shape[2]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axis_names = ["vx", "vy", "vz"]

    for j in range(3):
        ax = axes[0, j]
        ax.plot(t, v_des[:, j], color="black", linestyle=":", linewidth=1.2, label=r"$v_{des}$")
        ax.plot(t, v_m[:, j],   color="C0",    linestyle="--", linewidth=1.6, label=r"$v_m$ (ref)")
        ax.plot(t, v[:, j],     color="C3",    linewidth=2.0, label=r"$v$ (plant)")
        ax.set_title(f"{axis_names[j]} tracking")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("[m/s]")
        ax.legend(fontsize=8); ax.grid(True, linestyle="--", alpha=0.5)

        ax = axes[1, j]
        labels = [r"$\hat\Theta_0$ (bias)",
                  r"$\hat\Theta_1$ ($v$-coef)",
                  r"$\hat\Theta_2$ ($|v|v$-coef)"][:n_basis]
        for k in range(n_basis):
            ax.plot(t, theta[:, j, k], linewidth=1.6, label=labels[k])
        ax.set_title(f"{axis_names[j]} adaptive parameters")
        ax.set_xlabel("Time [s]")
        ax.legend(fontsize=8); ax.grid(True, linestyle="--", alpha=0.5)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)

    fig.suptitle("Figure 8 — Per-axis MRAC adaptation: reference-model tracking and parameter evolution",
                 fontsize=13, y=1.00)
    plt.tight_layout()
    _save(fig, "08_adaptation.png")


# ============================================================================
# Figure 09: control signals (motor thrusts)
# ============================================================================
def fig_09_control_signals(data_pid, data_mrac):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for k, (data, name, alpha, lw) in enumerate([
        (data_pid,  "PID",  0.85, 1.5),
        (data_mrac, "MRAC", 1.0,  2.0),
    ]):
        ax = axes[k]
        for i in range(4):
            ax.plot(data["t"], data["controls"][:, i],
                    color=f"C{i}", linewidth=lw, alpha=alpha,
                    label=f"u{i+1}")
        ax.axhline(5.0, color="red", linestyle="--", linewidth=1.0, alpha=0.6,
                   label="u_max")
        ax.set_ylabel(f"{name}\nmotor thrust [N]")
        ax.legend(loc="upper right", fontsize=9, ncol=5)
        ax.grid(True, linestyle="--", alpha=0.5)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Figure 9 — Per-motor thrusts (saturation at 5 N visible during aggressive maneuvers)",
                 fontsize=13, y=1.00)
    plt.tight_layout()
    _save(fig, "09_control_signals.png")


# ============================================================================
# Main
# ============================================================================
def run_scenario(seed=42, wind_func=None,
                 stop_tolerance=0.15, t_max=25.0, dt=0.005):
    """Run PID and MRAC on one randomly-generated scenario."""
    rng = np.random.default_rng(seed)
    while True:
        s = rng.uniform(5, 15, size=3); g = rng.uniform(5, 15, size=3)
        if np.linalg.norm(g - s) >= 6:
            break
    print(f"Scenario seed={seed}:  start={s}, target={g}, dist={np.linalg.norm(g-s):.2f} m")

    if wind_func is None:
        wind_func = _make_wind()

    system = QuadcopterSystem(c_drag_lin=0.22, c_drag_quad=0.10)
    x0 = np.zeros(12); x0[0:3] = s

    print("Running PID baseline...")
    ctrl_pid = Controller(**_kw_from_system(system))
    data_pid = run_simulation(system, ctrl_pid, wind_func, x0, g,
                              t_max=t_max, dt=dt, stop_tolerance=stop_tolerance,
                              stop_speed=None, verbose=False)
    print(f"  PID  t_end={data_pid['t'][-1]:.2f}s "
          f"final_err={np.linalg.norm(data_pid['states'][-1, 0:3] - g):.3f} m")

    print("Running MRAC + sigma-mod...")
    ctrl_mrac = MRACController(**_kw_from_system(system),
                               a_m_xy=6.0, a_m_z=8.0,
                               gamma=3.0, sigma=0.5,
                               theta_max=8.0, n_basis=3)
    data_mrac = run_simulation(system, ctrl_mrac, wind_func, x0, g,
                               t_max=t_max, dt=dt, stop_tolerance=stop_tolerance,
                               stop_speed=None, verbose=False)
    print(f"  MRAC t_end={data_mrac['t'][-1]:.2f}s "
          f"final_err={np.linalg.norm(data_mrac['states'][-1, 0:3] - g):.3f} m")

    return system, ctrl_pid, ctrl_mrac, data_pid, data_mrac, g, wind_func


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    system, ctrl_pid, ctrl_mrac, data_pid, data_mrac, target, wind_func = \
        run_scenario(seed=args.seed)

    print("\nGenerating figures...")
    fig_01_trajectory_3d(data_pid, data_mrac, target)
    fig_02_xy_topdown(data_pid, data_mrac, target)
    fig_03_state_signals(data_pid, data_mrac, target)
    fig_04_lyapunov(ctrl_mrac, gamma=3.0)
    fig_05_error_metrics(data_pid, data_mrac, target, eps=0.15)
    fig_06_phase_portrait(data_pid, data_mrac, target)
    fig_07_wind_estimation(ctrl_mrac, system, data_mrac, wind_func)
    fig_08_adaptation(ctrl_mrac)
    fig_09_control_signals(data_pid, data_mrac)
    print(f"\nAll figures saved to {_FIG_DIR}")


if __name__ == "__main__":
    main()
