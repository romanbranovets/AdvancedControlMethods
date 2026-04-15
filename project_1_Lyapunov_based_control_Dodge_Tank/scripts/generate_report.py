"""
scripts/generate_report.py
==========================
Generates all static figures and two separate animations for the project report.

Produces
--------
figures/01_trajectory.png        2-D path overview, colour-coded by time
figures/02_state_trajectories.png  x(t), y(t), theta(t)
figures/03_control_signals.png   track speeds and (v, omega)
figures/04_lyapunov.png          V(t) annotated with controller modes
figures/05_error_metrics.png     rho(t) and |alpha|(t)
figures/06_phase_portrait.png    V vs rho phase plot
animations/robot.mp4             robot animation (2-D field, silent)
animations/lyapunov.mp4          Lyapunov V(t) evolving animation

Usage
-----
  python scripts/generate_report.py            # seed=0, standard scenario
  python scripts/generate_report.py --seed 7   # different random scenario
  python scripts/generate_report.py --gif      # save .gif instead of .mp4
"""

from __future__ import annotations

import argparse
import sys
import os
import dataclasses
import warnings

# ── Make sure project root is on the path when run as a script ─────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")                          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from configs import DEFAULT_CONFIG as CFG
from main import run, generate_valid_random_scenario, random_config_with_seed
from src.visualization import _compute_lyapunov

# ── Output directories ──────────────────────────────────────────────────────
_FIG_DIR  = os.path.join(_PROJECT_ROOT, "figures")
_ANI_DIR  = os.path.join(_PROJECT_ROOT, "animations")
os.makedirs(_FIG_DIR, exist_ok=True)
os.makedirs(_ANI_DIR, exist_ok=True)

# ── Matplotlib style ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 130,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

# ── Colour palette for controller modes ────────────────────────────────────
MODE_COLOR = {
    "goal":              "#2979ff",
    "obstacle_avoidance":"#ab47bc",
    "dodge":             "#ff9800",
}
MODE_LABEL = {
    "goal":              "Goal tracking",
    "obstacle_avoidance":"Obstacle avoidance",
    "dodge":             "Projectile dodge",
}

# ── Helper: per-step mode colour array (aligns with controls / times[:-1]) ─
def _mode_colors(modes: tuple[str, ...]) -> list[str]:
    return [MODE_COLOR.get(m, "#888888") for m in modes]


def _save(fig: plt.Figure, name: str) -> None:
    path = os.path.join(_FIG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  STATIC FIGURES
# ═══════════════════════════════════════════════════════════════════════════

def plot_trajectory(sim, goal, obstacles, cannon_pos, lyapunov_c) -> None:
    """Fig 1 — 2-D trajectory on the planning field, colour-coded by time."""
    hist = sim.history
    times = sim.times
    modes = sim.modes
    N = len(hist) - 1

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_aspect("equal")
    ax.set_xlim(sim.xlim)
    ax.set_ylim(sim.ylim)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlabel("x, m")
    ax.set_ylabel("y, m")
    ax.set_title(
        "Figure 1 — Robot Trajectory on the Planning Field\n"
        "Path colour indicates time: dark blue = start, yellow = end."
    )

    # Static obstacles
    for i, (ox, oy, r) in enumerate(obstacles):
        ax.add_patch(Circle((ox, oy), r, facecolor="gray", edgecolor="black",
                             alpha=0.35, linewidth=1.5,
                             label="Obstacle" if i == 0 else None))

    # Cannon
    if cannon_pos is not None:
        ax.add_patch(Circle(cannon_pos, 0.45, facecolor="#1a1a2e",
                             edgecolor="#e94560", linewidth=2.5, zorder=8,
                             label="Cannon"))

    # Path coloured by time (LineCollection)
    xy = hist[:, :2]
    segs = np.stack([xy[:-1], xy[1:]], axis=1)
    lc = LineCollection(segs, cmap="plasma", linewidth=2.5, zorder=4, alpha=0.9)
    lc.set_array(times[:-1])
    ax.add_collection(lc)
    cb = fig.colorbar(lc, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Time, s")

    # Dodge segments highlighted separately
    dodge_segs = [seg for seg, m in zip(segs, modes) if m == "dodge"]
    if dodge_segs:
        lc_dodge = LineCollection(dodge_segs, color="#ff9800",
                                   linewidth=4, zorder=5, alpha=0.7,
                                   label="Dodge segment")
        ax.add_collection(lc_dodge)

    # Markers
    ax.plot(*hist[0, :2], "go", markersize=14, zorder=10, label="Start")
    goal_r = CFG.simulation.goal_tolerance
    ax.add_patch(Circle(goal, goal_r, facecolor="#ff000033", edgecolor="red",
                        linestyle="--", linewidth=2.0, zorder=5, label="Goal zone"))
    ax.plot(*goal, "r*", markersize=18, zorder=6)

    # Collision hits
    hits = [i for i, h in enumerate(sim.collision_steps) if h]
    if hits:
        hit_xy = hist[hits, :2]
        ax.scatter(hit_xy[:, 0], hit_xy[:, 1], color="red", s=80,
                   zorder=11, label=f"Hit ({len(hits)})")

    ax.legend(loc="upper left")
    fig.text(0.5, -0.02,
             "The robot navigates from start (green) to goal (red star) around "
             "obstacles (grey circles). Orange segments show projectile-dodge "
             "corrections — the robot barely deviates from the nominal path.",
             ha="center", fontsize=10, wrap=True)
    _save(fig, "01_trajectory.png")


def plot_state_trajectories(sim, goal) -> None:
    """Fig 2 — x(t), y(t), theta(t) state histories."""
    hist  = sim.history
    times = sim.times
    modes = sim.modes

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(
        "Figure 2 — Robot State Over Time\n"
        "All three degrees of freedom versus simulation time.",
        fontsize=13,
    )

    labels = ["x, m", "y, m", r"$\theta$, rad"]
    refs   = [goal[0], goal[1], None]
    colors = ["#1565c0", "#2e7d32", "#6a1b9a"]

    for ax, col, lbl, ref, data in zip(
        axes, colors, labels, refs, [hist[:, 0], hist[:, 1], hist[:, 2]]
    ):
        ax.plot(times, data, color=col, linewidth=1.8)
        if ref is not None:
            ax.axhline(ref, color="red", linestyle="--", linewidth=1.2,
                       label=f"Goal = {ref:.1f}")
            ax.legend(fontsize=9, loc="upper right")
        ax.set_ylabel(lbl)
        ax.grid(True, linestyle="--", alpha=0.5)

        # Shade dodge / avoidance periods
        t_step = times[1] - times[0]
        for i, m in enumerate(modes):
            if m != "goal" and i < len(times) - 1:
                ax.axvspan(times[i], times[i + 1],
                           alpha=0.12, color=MODE_COLOR.get(m, "gray"),
                           linewidth=0)

    axes[-1].set_xlabel("Time, s")

    # Mode legend
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.4, label=MODE_LABEL[k])
               for k, c in MODE_COLOR.items()]
    axes[0].legend(handles=handles, loc="upper right", fontsize=9)

    fig.text(0.5, -0.01,
             "Coloured bands: orange = dodge, purple = obstacle avoidance. "
             "x and y converge to the goal coordinates (red dashed).",
             ha="center", fontsize=10)
    plt.tight_layout()
    _save(fig, "02_state_trajectories.png")


def plot_control_signals(sim) -> None:
    """Fig 3 — track speeds (v_L, v_R) and unicycle (v, omega)."""
    controls = sim.controls    # (N, 2)
    times    = sim.times[:-1]  # (N,) — one per control step
    modes    = sim.modes
    L        = sim.track_center_distance
    u_max    = CFG.controller.u_max

    v_L = controls[:, 0]
    v_R = controls[:, 1]
    v   = 0.5 * (v_L + v_R)
    w   = (v_R - v_L) / L

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(
        "Figure 3 — Control Signals Over Time\n"
        "Top: track speeds sent to the robot.  Bottom: equivalent unicycle commands.",
        fontsize=13,
    )

    # Track speeds
    axes[0].plot(times, v_L, label="$v_L$ (left)",  linewidth=1.6, color="#1565c0")
    axes[0].plot(times, v_R, label="$v_R$ (right)", linewidth=1.6, color="#c62828",
                 alpha=0.8)
    if u_max is not None:
        axes[0].axhline( u_max, linestyle="--", color="black", linewidth=1.0,
                         label=f"$\\pm u_{{\\max}}$ = {u_max}")
        axes[0].axhline(-u_max, linestyle="--", color="black", linewidth=1.0)
    axes[0].set_ylabel("Track speed, m/s")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Unicycle
    ax2 = axes[1]
    ax2b = ax2.twinx()
    ax2.plot(times, v, label="$v$ (linear)", linewidth=1.6, color="#2e7d32")
    ax2b.plot(times, w, label=r"$\omega$ (angular)", linewidth=1.6,
              color="#e65100", alpha=0.8)
    ax2.set_ylabel("Linear speed $v$, m/s", color="#2e7d32")
    ax2b.set_ylabel(r"Angular speed $\omega$, rad/s", color="#e65100")
    ax2.tick_params(axis="y", colors="#2e7d32")
    ax2b.tick_params(axis="y", colors="#e65100")
    ax2.grid(True, linestyle="--", alpha=0.5)

    lines1, lbls1 = ax2.get_legend_handles_labels()
    lines2, lbls2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, lbls1 + lbls2, loc="upper right")
    axes[-1].set_xlabel("Time, s")

    # Shade modes
    for ax in axes:
        for i, m in enumerate(modes):
            if m != "goal" and i < len(times):
                t0 = times[i]
                t1 = times[i] + (times[1] - times[0]) if i + 1 < len(times) else t0
                ax.axvspan(t0, t1, alpha=0.12, color=MODE_COLOR.get(m, "gray"),
                           linewidth=0)

    fig.text(0.5, -0.01,
             "During obstacle avoidance and dodge, the track speed differential "
             r"($v_R - v_L$) increases to steer the robot; peak $\omega$ coincides "
             "with mode transitions.",
             ha="center", fontsize=10)
    plt.tight_layout()
    _save(fig, "03_control_signals.png")


def plot_lyapunov(sim, goal, lyapunov_c) -> None:
    """Fig 4 — Lyapunov function V(t) with mode annotations and hit markers."""
    hist   = sim.history
    times  = sim.times
    modes  = sim.modes
    V      = _compute_lyapunov(hist, np.asarray(goal), lyapunov_c)
    hits   = [i for i, h in enumerate(sim.collision_steps) if h]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(
        r"Figure 4 — Lyapunov Function $V(\rho,\alpha) = \frac{1}{2}\rho^2 + c(1-\cos\alpha)$"
        "\nV should decrease monotonically during goal-tracking; brief rises "
        "coincide with dodge / avoidance events.",
        fontsize=12,
    )

    # Mode shading
    for i, m in enumerate(modes):
        if m != "goal" and i < len(times) - 1:
            ax.axvspan(times[i], times[i + 1], alpha=0.2,
                       color=MODE_COLOR.get(m, "gray"), linewidth=0)

    # Hit markers
    for h in hits:
        if h < len(times):
            ax.axvline(times[h], color="red", linewidth=1.0, alpha=0.5)

    # V(t) curve, coloured by mode segment
    for i, m in enumerate(modes):
        if i + 1 < len(times):
            ax.plot(times[i:i+2], V[i:i+2],
                    color=MODE_COLOR.get(m, "#888888"), linewidth=2.0)

    ax.set_xlabel("Time, s")
    ax.set_ylabel(r"$V(\rho,\alpha)$")
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Legend
    handles = [
        Line2D([0],[0], color=c, linewidth=2, label=MODE_LABEL[k])
        for k, c in MODE_COLOR.items()
    ]
    handles += [
        plt.Rectangle((0,0),1,1, fc=MODE_COLOR["dodge"],   alpha=0.3, label="Dodge zone"),
        plt.Rectangle((0,0),1,1, fc=MODE_COLOR["obstacle_avoidance"], alpha=0.3,
                      label="Avoidance zone"),
        Line2D([0],[0], color="red", linewidth=1, label="Hit"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9)

    fig.text(0.5, -0.03,
             "V decreases monotonically in goal-tracking mode (blue). "
             "Brief increases (orange/purple zones) occur during dodge and "
             "obstacle avoidance; V resumes decreasing immediately after "
             "the manoeuvre completes, confirming asymptotic stability of the goal.",
             ha="center", fontsize=10)
    plt.tight_layout()
    _save(fig, "04_lyapunov.png")


def plot_error_metrics(sim, goal, lyapunov_c) -> None:
    """Fig 5 — rho(t) (distance to goal) and |alpha(t)| (heading error)."""
    hist   = sim.history
    times  = sim.times
    modes  = sim.modes
    goal_xy = np.asarray(goal, dtype=float)

    dx    = goal_xy[0] - hist[:, 0]
    dy    = goal_xy[1] - hist[:, 1]
    rho   = np.hypot(dx, dy)
    phi   = np.arctan2(dy, dx)
    alpha = np.angle(np.exp(1j * (phi - hist[:, 2])))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(
        "Figure 5 — Goal-Tracking Error Over Time\n"
        r"$\rho$ = distance to goal;   $|\alpha|$ = absolute heading error",
        fontsize=13,
    )

    axes[0].plot(times, rho, linewidth=1.8, color="#1565c0", label=r"$\rho(t)$, m")
    axes[0].axhline(CFG.simulation.goal_tolerance, linestyle="--", color="red",
                    linewidth=1.2, label=f"Tolerance = {CFG.simulation.goal_tolerance} m")
    axes[0].set_ylabel(r"$\rho$, m")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(times, np.abs(alpha), linewidth=1.8, color="#6a1b9a",
                 label=r"$|\alpha(t)|$, rad")
    axes[1].axhline(np.pi / 2, linestyle=":", color="gray", linewidth=1.0,
                    label=r"$\pi/2$")
    axes[1].set_ylabel(r"$|\alpha|$, rad")
    axes[1].set_xlabel("Time, s")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    for ax in axes:
        for i, m in enumerate(modes):
            if m != "goal" and i < len(times) - 1:
                ax.axvspan(times[i], times[i + 1], alpha=0.12,
                           color=MODE_COLOR.get(m, "gray"), linewidth=0)

    fig.text(0.5, -0.01,
             r"$\rho$ converges to zero as the robot reaches the goal. "
             r"$|\alpha|$ spikes during turns (obstacle avoidance, dodge) "
             "but returns near zero on straight segments.",
             ha="center", fontsize=10)
    plt.tight_layout()
    _save(fig, "05_error_metrics.png")


def plot_phase_portrait(sim, goal, lyapunov_c) -> None:
    """Fig 6 — Phase portrait: V vs rho, colour-coded by time."""
    hist    = sim.history
    times   = sim.times
    goal_xy = np.asarray(goal, dtype=float)
    V       = _compute_lyapunov(hist, goal_xy, lyapunov_c)
    rho     = np.hypot(goal_xy[0] - hist[:, 0], goal_xy[1] - hist[:, 1])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_title(
        r"Figure 6 — Phase Portrait: $V$ vs $\rho$"
        "\nFor a perfect Lyapunov descent each step moves left and down.",
        fontsize=12,
    )

    # Scatter coloured by time
    sc = ax.scatter(rho, V, c=times, cmap="plasma", s=6, alpha=0.7, zorder=3)
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Time, s")

    # Reference: V = 0.5*rho^2 (alpha=0 line — minimum possible V for given rho)
    rho_ref = np.linspace(0, rho.max() * 1.02, 200)
    ax.plot(rho_ref, 0.5 * rho_ref**2, "k--", linewidth=1.2,
            label=r"$V = \frac{1}{2}\rho^2$ ($\alpha=0$, lower bound)")

    ax.set_xlabel(r"$\rho$ (distance to goal), m")
    ax.set_ylabel(r"$V(\rho,\alpha)$")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9)

    fig.text(0.5, -0.02,
             "Points cluster near the lower bound curve (α ≈ 0) during "
             "straight-line goal tracking. Vertical scattering at fixed ρ "
             "reflects heading errors during turns.",
             ha="center", fontsize=10)
    plt.tight_layout()
    _save(fig, "06_phase_portrait.png")


# ═══════════════════════════════════════════════════════════════════════════
#  ANIMATIONS
# ═══════════════════════════════════════════════════════════════════════════

def _pick_writer(use_gif: bool):
    """Return (writer_instance, extension)."""
    if use_gif:
        return PillowWriter(fps=20), ".gif"
    try:
        import subprocess
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return FFMpegWriter(fps=25, codec="libx264",
                            extra_args=["-crf", "22", "-preset", "fast"]), ".mp4"
    except Exception:
        warnings.warn(
            "ffmpeg not found — falling back to Pillow GIF writer.",
            RuntimeWarning, stacklevel=2,
        )
        return PillowWriter(fps=20), ".gif"


def save_robot_animation(sim, goal, obstacles, cannon_pos, robot_radius,
                         lyapunov_c, path: str) -> None:
    """Animated 2-D field: robot, obstacles, projectiles, hitbox."""
    hist   = sim.history
    times  = sim.times
    modes  = sim.modes
    controls = sim.controls

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_xlim(sim.xlim)
    ax.set_ylim(sim.ylim)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlabel("x, m");  ax.set_ylabel("y, m")

    # Static: obstacles
    for i, (ox, oy, r) in enumerate(obstacles):
        ax.add_patch(Circle((ox, oy), r, facecolor="gray", edgecolor="black",
                             alpha=0.35, linewidth=1.5,
                             label="Obstacle" if i == 0 else None))
    # Static: cannon
    if cannon_pos is not None:
        ax.add_patch(Circle(cannon_pos, 0.45, facecolor="#1a1a2e",
                             edgecolor="#e94560", linewidth=2.5, zorder=8,
                             label="Cannon"))
        ax.plot(*cannon_pos, "r^", markersize=12, zorder=9)

    ax.plot(*hist[0, :2], "go", markersize=12, label="Start", zorder=5)
    goal_r = CFG.simulation.goal_tolerance
    ax.add_patch(Circle(goal, goal_r, facecolor="#ff000033", edgecolor="red",
                        linestyle="--", linewidth=2.0, zorder=5, label="Goal zone"))
    ax.plot(*goal, "r*", markersize=18, zorder=6)
    (path_line,) = ax.plot([], [], "b-", linewidth=2.5, alpha=0.8,
                           label="Path", zorder=4)
    (heading_line,) = ax.plot([], [], color="red", linewidth=3, zorder=11)

    body_patch = Polygon(np.zeros((4, 2)), closed=True, facecolor="lightblue",
                          edgecolor="darkblue", linewidth=2, alpha=0.95, zorder=10)
    left_track  = Polygon(np.zeros((4, 2)), closed=True, facecolor="#222",
                           edgecolor="black", linewidth=1.5, alpha=0.9, zorder=9)
    right_track = Polygon(np.zeros((4, 2)), closed=True, facecolor="#222",
                           edgecolor="black", linewidth=1.5, alpha=0.9, zorder=9)
    for p in (left_track, right_track, body_patch):
        ax.add_patch(p)

    hitbox = Circle((0, 0), robot_radius, facecolor="none", edgecolor="cyan",
                    linestyle="--", linewidth=1.5, alpha=0.7, zorder=13)
    ax.add_patch(hitbox)

    proj_patches: list[Circle] = []
    ax.legend(loc="upper left", fontsize=9)

    def update(frame):
        state = hist[frame]
        x, y, theta = state

        bdy, lft, rgt = sim._body_polygons(state)
        body_patch.set_xy(bdy);  left_track.set_xy(lft);  right_track.set_xy(rgt)

        al = 0.5 * sim.body_length
        heading_line.set_data([x, x + al * np.cos(theta)],
                              [y, y + al * np.sin(theta)])
        path_line.set_data(hist[:frame+1, 0], hist[:frame+1, 1])
        hitbox.center = (x, y)

        is_hit = frame < len(sim.collision_steps) and sim.collision_steps[frame]
        if is_hit:
            hitbox.set_edgecolor("red");   hitbox.set_linewidth(3)
            body_patch.set_facecolor("#ff4444")
        else:
            hitbox.set_edgecolor("cyan");  hitbox.set_linewidth(1.5)
            body_patch.set_facecolor("lightblue")

        for p in proj_patches:
            p.remove()
        proj_patches.clear()
        if frame < len(sim.projectile_snapshots):
            for (px, py, pr) in sim.projectile_snapshots[frame]:
                c = Circle((px, py), pr, facecolor="#e94560", edgecolor="#f00",
                           alpha=0.9, linewidth=1.2, zorder=12)
                ax.add_patch(c);  proj_patches.append(c)

        mode_str = ""
        m = modes[min(frame, len(modes) - 1)] if modes else "goal"
        if is_hit:          mode_str = " | HIT"
        elif m == "dodge":  mode_str = " | DODGE"
        elif m == "obstacle_avoidance": mode_str = " | AVOIDANCE"

        t = times[frame]
        if frame < len(controls):
            vl, vr = controls[frame]
            ax.set_title(
                f"Robot animation   t = {t:.2f} s{mode_str}\n"
                f"$v_L$ = {vl:+.2f} m/s   $v_R$ = {vr:+.2f} m/s",
                fontsize=11,
            )
        else:
            ax.set_title(f"Robot animation   t = {t:.2f} s{mode_str}", fontsize=11)

        return path_line, body_patch, left_track, right_track, heading_line, hitbox

    ani = FuncAnimation(fig, update, frames=len(hist),
                        interval=max(1, int(1000 * sim.dt)), blit=False, repeat=False)
    writer, ext = _pick_writer(use_gif=path.endswith(".gif"))
    out = path if path.endswith(ext) else os.path.splitext(path)[0] + ext
    ani.save(out, writer=writer)
    plt.close(fig)
    print(f"  saved {out}")


def save_lyapunov_animation(sim, goal, lyapunov_c, path: str) -> None:
    """Animated Lyapunov V(t) curve growing frame by frame."""
    hist   = sim.history
    times  = sim.times
    modes  = sim.modes
    V_all  = _compute_lyapunov(hist, np.asarray(goal, dtype=float), lyapunov_c)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(0, V_all.max() * 1.08 + 0.5)
    ax.set_xlabel("Time, s")
    ax.set_ylabel(r"$V(\rho,\alpha)$")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Mode shading (static background)
    for i, m in enumerate(modes):
        if m != "goal" and i < len(times) - 1:
            ax.axvspan(times[i], times[i + 1], alpha=0.18,
                       color=MODE_COLOR.get(m, "gray"), linewidth=0)
    for i, hit in enumerate(sim.collision_steps):
        if hit and i < len(times) - 1:
            ax.axvspan(times[i], times[i + 1], alpha=0.35, color="red", linewidth=0)

    # Full trajectory (ghost)
    ax.plot(times, V_all, color="#cccccc", linewidth=1.0, zorder=2)

    (live_line,) = ax.plot([], [], color="#2979ff", linewidth=2.2, zorder=3,
                           label=r"$V(t)$ — current progress")
    vline = ax.axvline(0, color="#ff5252", linewidth=1.5, linestyle="--", zorder=4)
    val_text = ax.text(0.97, 0.96, "", transform=ax.transAxes,
                       ha="right", va="top", fontsize=11, color="#2979ff",
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    handles = [
        Line2D([0],[0], color="#2979ff", linewidth=2, label=r"$V(t)$"),
        plt.Rectangle((0,0),1,1, fc=MODE_COLOR["dodge"], alpha=0.4,
                      label="Dodge"),
        plt.Rectangle((0,0),1,1, fc=MODE_COLOR["obstacle_avoidance"], alpha=0.4,
                      label="Avoidance"),
        plt.Rectangle((0,0),1,1, fc="red", alpha=0.5, label="Hit"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=10)

    def update(frame):
        live_line.set_data(times[:frame+1], V_all[:frame+1])
        vline.set_xdata([times[frame], times[frame]])
        val_text.set_text(f"V = {V_all[frame]:.3f}")
        m = modes[min(frame, len(modes)-1)] if modes else "goal"
        hit = frame < len(sim.collision_steps) and sim.collision_steps[frame]
        mode_str = " | HIT" if hit else (" | DODGE" if m == "dodge" else
                    (" | AVOIDANCE" if m == "obstacle_avoidance" else ""))
        ax.set_title(
            r"Lyapunov function  $V = \frac{1}{2}\rho^2 + c(1{-}\cos\alpha)$"
            f"\nt = {times[frame]:.2f} s{mode_str}",
            fontsize=12,
        )
        return live_line, vline, val_text

    ani = FuncAnimation(fig, update, frames=len(hist),
                        interval=max(1, int(1000 * sim.dt)), blit=False, repeat=False)
    writer, ext = _pick_writer(use_gif=path.endswith(".gif"))
    out = path if path.endswith(ext) else os.path.splitext(path)[0] + ext
    ani.save(out, writer=writer)
    plt.close(fig)
    print(f"  saved {out}")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate all report figures and animations."
    )
    parser.add_argument("--seed", type=int, default=0,
                        help="Random scenario seed (default: 0)")
    parser.add_argument("--steps", type=int, default=CFG.simulation.num_steps,
                        help="Max simulation steps")
    parser.add_argument("--no-cannon", action="store_true",
                        help="Disable cannon (clean Lyapunov descent)")
    parser.add_argument("--gif", action="store_true",
                        help="Save animations as .gif instead of .mp4")
    args = parser.parse_args()

    goal = CFG.goal
    cannon_config = dataclasses.replace(CFG.cannon, enabled=not args.no_cannon,
                                        seed=args.seed + 1)

    # Generate a validated random scenario
    print(f"[1/2] Generating scenario (seed={args.seed}) ...")
    random_cfg = random_config_with_seed(CFG.random, args.seed)
    initial_state, obstacles = generate_valid_random_scenario(goal=goal,
                                                               config=random_cfg)
    print(f"      start={tuple(round(v, 2) for v in initial_state)}, "
          f"obstacles={len(obstacles)}")

    print("[2/2] Running simulation ...")
    sim = run(
        num_steps=args.steps,
        initial_state=initial_state,
        goal=goal,
        obstacles=obstacles,
        noise_config=CFG.noise,
        cannon_config=cannon_config,
        render=False,
    )
    robot_radius  = CFG.controller.robot_radius
    lyapunov_c    = CFG.controller.lyapunov_c
    cannon_pos    = (CFG.cannon.x, CFG.cannon.y) if not args.no_cannon else None
    final_err     = float(np.linalg.norm(sim.pose[:2] - np.asarray(goal)))
    print(f"      steps={len(sim.history)-1}  hits={sim.hit_count}  "
          f"goal_error={final_err:.4f} m")

    # ── Static figures ──────────────────────────────────────────────────────
    print("\nGenerating static figures ...")
    plot_trajectory(sim, goal, obstacles, cannon_pos, lyapunov_c)
    plot_state_trajectories(sim, goal)
    plot_control_signals(sim)
    plot_lyapunov(sim, goal, lyapunov_c)
    plot_error_metrics(sim, goal, lyapunov_c)
    plot_phase_portrait(sim, goal, lyapunov_c)

    # ── Animations ─────────────────────────────────────────────────────────
    ext = ".gif" if args.gif else ".mp4"
    print("\nRendering animations (this may take a minute) ...")

    save_robot_animation(
        sim, goal, obstacles, cannon_pos, robot_radius, lyapunov_c,
        path=os.path.join(_ANI_DIR, f"robot{ext}"),
    )
    save_lyapunov_animation(
        sim, goal, lyapunov_c,
        path=os.path.join(_ANI_DIR, f"lyapunov{ext}"),
    )

    print("\nDone.  Output summary:")
    print(f"  figures/  → {_FIG_DIR}")
    print(f"  animations/ → {_ANI_DIR}")


if __name__ == "__main__":
    main()
