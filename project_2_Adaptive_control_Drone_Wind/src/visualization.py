# src/visualization.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def euler_to_rotation_matrix(euler):
    phi, theta, psi = euler
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth,  sth  = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi),  np.sin(psi)
    R = np.array([
        [cpsi*cth,  cpsi*sth*sphi - spsi*cphi,  cpsi*sth*cphi + spsi*sphi],
        [spsi*cth,  spsi*sth*sphi + cpsi*cphi,  spsi*sth*cphi - cpsi*sphi],
        [-sth,      cth*sphi,                    cth*cphi]
    ])
    return R


def visualize(data, target_fps=30, show=True):
    """3D animation. Subsamples frames so playback matches wall-clock.

    target_fps: desired animation FPS; stride is derived from sim dt.
    """
    t = data['t']
    states = data['states']
    controls = data['controls']
    winds = data['winds']
    target = data['target']

    # derive simulation dt and downsample to target_fps
    if len(t) > 1:
        dt_sim = float(np.median(np.diff(t)))
    else:
        dt_sim = 0.005
    stride = max(1, int(round(1.0 / (target_fps * dt_sim))))
    idx = np.arange(0, len(t), stride)
    if idx[-1] != len(t) - 1:
        idx = np.append(idx, len(t) - 1)

    pos_data = states[:, 0:3]
    finite_mask = np.isfinite(pos_data).all(axis=1)
    if not finite_mask.any():
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.text(0, 0, 0, "Simulation diverged", color='red', fontsize=14, ha='center')
        plt.show()
        return

    finite_pos = pos_data[finite_mask]
    xmin, ymin, zmin = finite_pos.min(axis=0)
    xmax, ymax, zmax = finite_pos.max(axis=0)
    pad = 2.0

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_zlim(max(0.0, zmin - 1.0), zmax + pad)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    # static artists (drawn once)
    ax.scatter(target[0], target[1], target[2], color='red', s=180, marker='*',
               label=f'Target ({target[0]:.1f},{target[1]:.1f},{target[2]:.1f})')
    ax.scatter(pos_data[0, 0], pos_data[0, 1], pos_data[0, 2],
               color='lime', s=120, marker='o', label='Start')
    path_line, = ax.plot([], [], [], 'b-', linewidth=2.0, label='Path')
    title = ax.set_title('')
    ax.legend(loc='upper left', fontsize=9)

    visual_l = 0.35
    motor_offsets_body = np.array([[ visual_l, 0, 0],
                                   [ 0,  visual_l, 0],
                                   [-visual_l, 0, 0],
                                   [ 0, -visual_l, 0]])

    arm_lines = [ax.plot([], [], [], color='black', linewidth=5)[0] for _ in range(4)]
    thrust_quivers = [None] * 4
    wind_quiver = [None]

    interval_ms = 1000.0 / target_fps

    def _update(frame_idx):
        f = idx[frame_idx]
        state = states[f]
        pos = state[0:3]
        euler = state[6:9]
        u = controls[f]
        wind = winds[f]
        R = euler_to_rotation_matrix(euler)

        # path
        sub = idx[:frame_idx + 1]
        path_line.set_data(pos_data[sub, 0], pos_data[sub, 1])
        path_line.set_3d_properties(pos_data[sub, 2])

        # arms
        motor_pos = pos + (R @ motor_offsets_body.T).T
        for i in range(4):
            arm_lines[i].set_data([pos[0], motor_pos[i, 0]],
                                  [pos[1], motor_pos[i, 1]])
            arm_lines[i].set_3d_properties([pos[2], motor_pos[i, 2]])

        def _safe_remove(artist):
            if artist is None:
                return
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass

        # refresh thrust & wind quivers (quivers must be recreated)
        for i in range(4):
            _safe_remove(thrust_quivers[i])
            thrust_world = R @ np.array([0.0, 0.0, u[i] * 0.12])
            thrust_quivers[i] = ax.quiver(motor_pos[i, 0], motor_pos[i, 1], motor_pos[i, 2],
                                          thrust_world[0], thrust_world[1], thrust_world[2],
                                          color='cyan', alpha=0.9, linewidth=2.0,
                                          arrow_length_ratio=0.35)

        _safe_remove(wind_quiver[0])
        wind_quiver[0] = None
        w_norm = np.linalg.norm(wind)
        if w_norm > 0.1:
            wind_scale = min(w_norm * 0.6, 4.0) / w_norm
            wind_pos = np.array([xmin - pad * 0.5, ymin - pad * 0.5, zmax + pad * 0.5])
            wind_quiver[0] = ax.quiver(wind_pos[0], wind_pos[1], wind_pos[2],
                                       wind[0] * wind_scale, wind[1] * wind_scale, wind[2] * wind_scale,
                                       color='blue', alpha=0.5, linewidth=3.0,
                                       arrow_length_ratio=0.35)

        title.set_text(f't = {t[f]:.2f} s  |  wind |v|={w_norm:.2f} m/s')
        return (path_line, title, *arm_lines)

    ani = FuncAnimation(fig, _update, frames=len(idx),
                        interval=interval_ms, blit=False, repeat=True)
    # keep a reference so GC doesn't kill the animation
    fig._ani_ref = ani
    if show:
        plt.show()
    return ani
