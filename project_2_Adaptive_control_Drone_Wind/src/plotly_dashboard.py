"""
src/plotly_dashboard.py
=======================
Interactive HTML dashboard with synchronized animation across 4 panels:
  - 3D rotatable trajectory (drag to rotate during playback)
  - Top-down 2D X vs Y
  - Cylindrical: sqrt((x-x_t)^2 + (y-y_t)^2) vs Z
  - Phase portrait: r vs r_dot (distance to target and its rate)

Static layer: full PID (translucent) and MRAC (solid) paths are drawn once.
Animated layer: only the "current position" markers move — keeps the HTML small.
"""

from __future__ import annotations

import numpy as np


def _pad(arr, n_target, fill_last=True):
    if len(arr) >= n_target:
        return arr[:n_target]
    pad = n_target - len(arr)
    if arr.ndim == 1:
        return np.concatenate([arr, np.full(pad, arr[-1] if fill_last else np.nan)])
    return np.concatenate([arr, np.tile(arr[-1] if fill_last else np.zeros_like(arr[0]),
                                        (pad, 1))])


def _r_and_rdot(states, target):
    pos = states[:, 0:3]; vel = states[:, 3:6]
    delta = pos - target
    r = np.linalg.norm(delta, axis=1)
    r_safe = np.where(r > 1e-6, r, 1e-6)
    r_dot = np.einsum("ij,ij->i", delta, vel) / r_safe
    return r, r_dot


def build_dashboard(data_pid, data_mrac, target,
                    label_pid="PID baseline", label_mrac="MRAC",
                    target_fps=10, save_path="dashboard.html",
                    title="Drone with wind: PID vs MRAC"):
    """
    Build a 4-panel interactive Plotly dashboard with synchronized animation.

    All four panels advance with one shared slider/play button. The 3D panel
    can be freely rotated/zoomed during playback. PID and MRAC full paths are
    drawn statically; only the moving cursors are animated (small HTML).
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as e:
        raise RuntimeError(
            "plotly is required for the interactive dashboard. "
            "Install with `pip install plotly`."
        ) from e

    target = np.asarray(target, dtype=float)

    # ---- align lengths ------------------------------------------------------
    n_max = max(len(data_pid["t"]), len(data_mrac["t"]))
    pid_t  = _pad(data_pid["t"],      n_max)
    pid_s  = _pad(data_pid["states"], n_max)
    mrac_s = _pad(data_mrac["states"], n_max)

    # ---- subsample to target_fps for animation cursors ----------------------
    dt_sim = float(np.median(np.diff(pid_t))) if n_max > 1 else 0.01
    stride = max(1, int(round(1.0 / (target_fps * dt_sim))))
    idx = np.arange(0, n_max, stride)
    if idx[-1] != n_max - 1:
        idx = np.append(idx, n_max - 1)

    # full-resolution paths for static layer
    p_pid  = pid_s[:, 0:3]
    p_mrac = mrac_s[:, 0:3]
    rho_xy_pid  = np.hypot(p_pid[:, 0]  - target[0], p_pid[:, 1]  - target[1])
    rho_xy_mrac = np.hypot(p_mrac[:, 0] - target[0], p_mrac[:, 1] - target[1])
    r_pid,  rdot_pid  = _r_and_rdot(pid_s,  target)
    r_mrac, rdot_mrac = _r_and_rdot(mrac_s, target)

    # ---- subplot grid -------------------------------------------------------
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "scatter3d", "rowspan": 2}, {"type": "xy"}],
               [None,                                 {"type": "xy"}]],
        column_widths=[0.55, 0.45],
        row_heights=[0.5, 0.5],
        subplot_titles=("3D trajectory  (drag to rotate)",
                        "Top-down  X vs Y",
                        "Phase portrait  r vs dr/dt"),
        horizontal_spacing=0.06,
        vertical_spacing=0.10,
    )

    # ---- bounds for 3D ------------------------------------------------------
    pos_all = np.vstack([p_pid, p_mrac, target.reshape(1, 3)])
    pad = 2.0
    xmin, ymin, zmin = pos_all.min(axis=0) - pad
    xmax, ymax, zmax = pos_all.max(axis=0) + pad
    zmin = max(0.0, zmin)

    GRAY      = "rgba(150,150,150,0.6)"
    GRAY_DOT  = "rgba(120,120,120,1.0)"
    BLUE      = "rgba(21,101,192,1.0)"
    BLUE_FILL = "rgba(21,101,192,0.20)"
    RED       = "rgba(211,47,47,1.0)"
    GREEN     = "rgba(46,125,50,1.0)"

    # =========================== STATIC LAYER ================================

    # 3D: full paths, target & start
    fig.add_trace(go.Scatter3d(
        x=p_pid[:, 0], y=p_pid[:, 1], z=p_pid[:, 2],
        mode="lines", line=dict(color=GRAY, width=4),
        name=label_pid, showlegend=True,
    ), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=p_mrac[:, 0], y=p_mrac[:, 1], z=p_mrac[:, 2],
        mode="lines", line=dict(color=BLUE, width=6),
        name=label_mrac, showlegend=True,
    ), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=[target[0]], y=[target[1]], z=[target[2]],
        mode="markers", marker=dict(size=8, color="red", symbol="diamond"),
        name="Target", showlegend=True,
    ), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=[p_pid[0, 0]], y=[p_pid[0, 1]], z=[p_pid[0, 2]],
        mode="markers", marker=dict(size=6, color="green"),
        name="Start", showlegend=True,
    ), row=1, col=1)

    # 2D XY
    fig.add_trace(go.Scatter(
        x=p_pid[:, 0],  y=p_pid[:, 1],
        mode="lines", line=dict(color=GRAY, width=2),
        showlegend=False, name=f"{label_pid} XY",
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=p_mrac[:, 0], y=p_mrac[:, 1],
        mode="lines", line=dict(color=BLUE, width=3),
        showlegend=False, name=f"{label_mrac} XY",
    ), row=1, col=2)
    # epsilon-ball circle (top-down disk of radius eps)
    eps = 0.15
    th = np.linspace(0, 2 * np.pi, 80)
    fig.add_trace(go.Scatter(
        x=target[0] + eps * np.cos(th),
        y=target[1] + eps * np.sin(th),
        mode="lines", line=dict(color=RED, dash="dash", width=1),
        fill="toself", fillcolor="rgba(211,47,47,0.10)",
        name=f"epsilon = {eps:.2f} m", showlegend=True,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[target[0]], y=[target[1]],
        mode="markers", marker=dict(size=14, color="red", symbol="star"),
        showlegend=False,
    ), row=1, col=2)

    # Phase portrait
    fig.add_trace(go.Scatter(
        x=r_pid, y=rdot_pid,
        mode="lines", line=dict(color=GRAY, width=2),
        showlegend=False, name=f"{label_pid} phase",
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=r_mrac, y=rdot_mrac,
        mode="lines", line=dict(color=BLUE, width=3),
        showlegend=False, name=f"{label_mrac} phase",
    ), row=2, col=2)

    # ===================== ANIMATED LAYER (cursors only) =====================

    # The next 6 traces are updated per-frame.
    # Order: 3D-pid-now, 3D-mrac-now, XY-pid-now, XY-mrac-now, phase-pid-now, phase-mrac-now
    cursor_indices = list(range(len(fig.data), len(fig.data) + 6))

    fig.add_trace(go.Scatter3d(
        x=[p_pid[0, 0]], y=[p_pid[0, 1]], z=[p_pid[0, 2]],
        mode="markers", marker=dict(size=7, color=GRAY_DOT),
        showlegend=False, name="pid_now_3d",
    ), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=[p_mrac[0, 0]], y=[p_mrac[0, 1]], z=[p_mrac[0, 2]],
        mode="markers", marker=dict(size=10, color=BLUE),
        showlegend=False, name="mrac_now_3d",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=[p_pid[0, 0]],  y=[p_pid[0, 1]],
        mode="markers", marker=dict(size=8, color=GRAY_DOT),
        showlegend=False, name="pid_now_xy",
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[p_mrac[0, 0]], y=[p_mrac[0, 1]],
        mode="markers", marker=dict(size=11, color=BLUE),
        showlegend=False, name="mrac_now_xy",
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=[r_pid[0]],  y=[rdot_pid[0]],
        mode="markers", marker=dict(size=8, color=GRAY_DOT),
        showlegend=False, name="pid_now_phase",
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=[r_mrac[0]], y=[rdot_mrac[0]],
        mode="markers", marker=dict(size=11, color=BLUE),
        showlegend=False, name="mrac_now_phase",
    ), row=2, col=2)

    # ---- frames -------------------------------------------------------------
    frames = []
    for k in idx:
        f_data = [
            go.Scatter3d(x=[p_pid[k, 0]],  y=[p_pid[k, 1]],  z=[p_pid[k, 2]]),
            go.Scatter3d(x=[p_mrac[k, 0]], y=[p_mrac[k, 1]], z=[p_mrac[k, 2]]),
            go.Scatter(x=[p_pid[k, 0]],  y=[p_pid[k, 1]]),
            go.Scatter(x=[p_mrac[k, 0]], y=[p_mrac[k, 1]]),
            go.Scatter(x=[r_pid[k]],  y=[rdot_pid[k]]),
            go.Scatter(x=[r_mrac[k]], y=[rdot_mrac[k]]),
        ]
        frames.append(go.Frame(data=f_data, name=f"{pid_t[k]:.2f}",
                               traces=cursor_indices))

    fig.frames = frames

    # ---- axes ---------------------------------------------------------------
    fig.update_scenes(
        xaxis=dict(title="X [m]", range=[xmin, xmax]),
        yaxis=dict(title="Y [m]", range=[ymin, ymax]),
        zaxis=dict(title="Z [m]", range=[zmin, zmax]),
        aspectmode="data",
    )
    fig.update_xaxes(title="X [m]", row=1, col=2,
                     range=[pos_all[:, 0].min() - 1, pos_all[:, 0].max() + 1])
    fig.update_yaxes(title="Y [m]", row=1, col=2, scaleanchor="x", scaleratio=1,
                     range=[pos_all[:, 1].min() - 1, pos_all[:, 1].max() + 1])
    rmax = float(max(r_pid.max(), r_mrac.max(), 1.0)) * 1.05
    rdot_min = float(min(rdot_pid.min(), rdot_mrac.min()))
    rdot_max = float(max(rdot_pid.max(), rdot_mrac.max()))
    fig.update_xaxes(title="r = ||p - p_t|| [m]", row=2, col=2, range=[0, rmax])
    fig.update_yaxes(title="dr/dt [m/s]", row=2, col=2,
                     range=[rdot_min - 0.5, rdot_max + 0.5])

    # ---- play / slider ------------------------------------------------------
    fig.update_layout(
        title=dict(
            text=(f"<b>{title}</b><br>"
                  f"<span style='font-size:12px;color:gray'>"
                  f"PID translucent gray, MRAC solid blue. "
                  f"Drag the 3D panel to rotate during playback.</span>"),
            x=0.5,
        ),
        updatemenus=[dict(
            type="buttons", showactive=False,
            x=0.5, y=-0.05, xanchor="center", yanchor="top",
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, dict(frame=dict(duration=int(1000/target_fps), redraw=True),
                                      fromcurrent=True, mode="immediate",
                                      transition=dict(duration=0))]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate")]),
            ],
        )],
        sliders=[dict(
            active=0, x=0.0, y=-0.10, len=1.0,
            currentvalue=dict(prefix="t = ", suffix=" s", font=dict(size=12)),
            steps=[dict(method="animate", label=f.name,
                        args=[[f.name], dict(mode="immediate",
                                             frame=dict(duration=0, redraw=True),
                                             transition=dict(duration=0))])
                   for f in frames],
        )],
        legend=dict(x=0.0, y=1.0, bgcolor="rgba(255,255,255,0.7)"),
        margin=dict(l=10, r=10, t=80, b=80),
        height=820,
    )

    fig.write_html(save_path, include_plotlyjs="cdn", auto_play=False)
    print(f"[plotly] dashboard saved to {save_path}  (open in browser)")
    return fig
