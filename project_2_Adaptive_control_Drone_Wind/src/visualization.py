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


def visualize(data, target_fps=30, show=True, save_path=None, save_dpi=120):
    """3D animation. Subsamples frames so playback matches wall-clock.

    target_fps: desired animation FPS; stride is derived from sim dt.
    save_path:  optional path to save animation. Extension picks writer:
                '.mp4'/'.mov'/'.avi' -> ffmpeg, '.gif' -> pillow.
    save_dpi:   resolution for saved file.
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

    if save_path is not None:
        _save_animation(ani, save_path, target_fps, save_dpi, n_frames=len(idx))

    if show:
        plt.show()
    return ani


# ============================================================================
# Side-by-side compare animation: baseline (translucent) + main (solid)
# ============================================================================

def _safe_remove(artist):
    if artist is None:
        return
    try:
        artist.remove()
    except (ValueError, AttributeError):
        pass


def _save_animation(ani, save_path, target_fps, save_dpi, n_frames):
    ext = save_path.lower().rsplit('.', 1)[-1]
    if ext in ('mp4', 'mov', 'avi', 'mkv'):
        writer = 'ffmpeg'
    else:
        writer = 'pillow'
    print(f"[viz] saving {n_frames} frames to {save_path} (writer={writer}, fps={target_fps})...")
    try:
        ani.save(save_path, writer=writer, fps=target_fps, dpi=save_dpi)
        print(f"[viz] saved {save_path}")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[viz] save failed ({e}). For .mp4 install ffmpeg; "
              f"for .gif use save_path ending in .gif.")


def _pad_to(data, n_target):
    """Repeat last frame so dataset length matches n_target."""
    n = len(data['t'])
    if n >= n_target:
        return data
    pad = n_target - n
    if n > 1:
        dt = float(np.median(np.diff(data['t'])))
    else:
        dt = 0.005
    t_pad = data['t'][-1] + dt * np.arange(1, pad + 1)
    return {
        't':        np.concatenate([data['t'], t_pad]),
        'states':   np.concatenate([data['states'],   np.tile(data['states'][-1],   (pad, 1))]),
        'controls': np.concatenate([data['controls'], np.tile(data['controls'][-1], (pad, 1))]),
        'winds':    np.concatenate([data['winds'],    np.tile(data['winds'][-1],    (pad, 1))]),
        'target':   data['target'],
    }


class _DroneArtists:
    """Per-drone matplotlib artists (path + 4 arms + 4 thrust quivers)."""

    _MOTOR_OFFSETS = np.array([[ 0.35,  0.0,  0.0],
                               [ 0.0,   0.35, 0.0],
                               [-0.35,  0.0,  0.0],
                               [ 0.0,  -0.35, 0.0]])

    def __init__(self, ax, *, label, alpha=1.0,
                 path_color='C0', arm_color='black', thrust_color='cyan',
                 path_lw=2.0, arm_lw=5.0):
        self.ax = ax
        self.alpha = alpha
        self.thrust_color = thrust_color
        self.path_line, = ax.plot([], [], [], '-', linewidth=path_lw,
                                  color=path_color, alpha=alpha, label=label)
        self.arm_lines = [ax.plot([], [], [], color=arm_color, linewidth=arm_lw,
                                  alpha=alpha)[0] for _ in range(4)]
        self.thrust_quivers = [None] * 4

    def update(self, frame_idx, idx, pos_data, states, controls):
        f = idx[frame_idx]
        state = states[f]
        pos   = state[0:3]
        euler = state[6:9]
        u     = controls[f]
        R = euler_to_rotation_matrix(euler)

        sub = idx[:frame_idx + 1]
        self.path_line.set_data(pos_data[sub, 0], pos_data[sub, 1])
        self.path_line.set_3d_properties(pos_data[sub, 2])

        motor_pos = pos + (R @ self._MOTOR_OFFSETS.T).T
        for i in range(4):
            self.arm_lines[i].set_data([pos[0], motor_pos[i, 0]],
                                       [pos[1], motor_pos[i, 1]])
            self.arm_lines[i].set_3d_properties([pos[2], motor_pos[i, 2]])

        for i in range(4):
            _safe_remove(self.thrust_quivers[i])
            tw = R @ np.array([0.0, 0.0, u[i] * 0.12])
            self.thrust_quivers[i] = self.ax.quiver(
                motor_pos[i, 0], motor_pos[i, 1], motor_pos[i, 2],
                tw[0], tw[1], tw[2],
                color=self.thrust_color, alpha=0.9 * self.alpha,
                linewidth=2.0, arrow_length_ratio=0.35,
            )


def visualize_compare(data_main, data_baseline,
                      label_main='MRAC', label_baseline='PID baseline',
                      target_fps=30, show=True,
                      save_path=None, save_dpi=120):
    """3D animation overlaying two simulation runs on one scene.

    The `data_baseline` drone is rendered translucent; `data_main` is opaque.
    Both runs share the same target and wind. Shorter run is padded with its
    last state so both finish together visually.
    """
    n_max = max(len(data_main['t']), len(data_baseline['t']))
    data_main     = _pad_to(data_main,     n_max)
    data_baseline = _pad_to(data_baseline, n_max)

    t = data_main['t']
    target = data_main['target']

    if len(t) > 1:
        dt_sim = float(np.median(np.diff(t)))
    else:
        dt_sim = 0.005
    stride = max(1, int(round(1.0 / (target_fps * dt_sim))))
    idx = np.arange(0, len(t), stride)
    if idx[-1] != len(t) - 1:
        idx = np.append(idx, len(t) - 1)

    pos_main = data_main['states'][:, 0:3]
    pos_base = data_baseline['states'][:, 0:3]
    all_pos = np.vstack([pos_main, pos_base, target.reshape(1, 3)])
    finite = np.isfinite(all_pos).all(axis=1)
    if not finite.any():
        print("[viz_compare] no finite positions, aborting"); return
    pos_finite = all_pos[finite]
    xmin, ymin, zmin = pos_finite.min(axis=0)
    xmax, ymax, zmax = pos_finite.max(axis=0)
    pad = 2.0

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_zlim(max(0.0, zmin - 1.0), zmax + pad)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    ax.scatter(target[0], target[1], target[2], color='red', s=180, marker='*',
               label=f'Target ({target[0]:.1f},{target[1]:.1f},{target[2]:.1f})')
    ax.scatter(pos_main[0, 0], pos_main[0, 1], pos_main[0, 2],
               color='lime', s=120, marker='o', label='Start')

    drone_base = _DroneArtists(ax, label=label_baseline, alpha=0.32,
                               path_color='gray', arm_color='gray',
                               thrust_color='gray',
                               path_lw=1.6, arm_lw=3.5)
    drone_main = _DroneArtists(ax, label=label_main, alpha=1.0,
                               path_color='tab:blue', arm_color='black',
                               thrust_color='cyan',
                               path_lw=2.4, arm_lw=5.5)

    title = ax.set_title('')
    ax.legend(loc='upper left', fontsize=9)
    wind_quiver = [None]

    interval_ms = 1000.0 / target_fps

    def _update(frame_idx):
        drone_base.update(frame_idx, idx, pos_base,
                          data_baseline['states'], data_baseline['controls'])
        drone_main.update(frame_idx, idx, pos_main,
                          data_main['states'], data_main['controls'])

        f = idx[frame_idx]
        wind = data_main['winds'][f]
        _safe_remove(wind_quiver[0])
        wind_quiver[0] = None
        w_norm = float(np.linalg.norm(wind))
        if w_norm > 0.1:
            wind_scale = min(w_norm * 0.6, 4.0) / w_norm
            wp = np.array([xmin - pad * 0.5, ymin - pad * 0.5, zmax + pad * 0.5])
            wind_quiver[0] = ax.quiver(wp[0], wp[1], wp[2],
                                       wind[0] * wind_scale,
                                       wind[1] * wind_scale,
                                       wind[2] * wind_scale,
                                       color='blue', alpha=0.5, linewidth=3.0,
                                       arrow_length_ratio=0.35)

        err_main = np.linalg.norm(pos_main[f] - target)
        err_base = np.linalg.norm(pos_base[f] - target)
        title.set_text(
            f't = {t[f]:.2f} s   |   '
            f'{label_main}: err = {err_main:.2f} m   |   '
            f'{label_baseline}: err = {err_base:.2f} m   |   '
            f'wind |v|={w_norm:.2f} m/s'
        )

    ani = FuncAnimation(fig, _update, frames=len(idx),
                        interval=interval_ms, blit=False, repeat=True)
    fig._ani_ref = ani

    if save_path is not None:
        _save_animation(ani, save_path, target_fps, save_dpi, n_frames=len(idx))

    if show:
        plt.show()
    return ani
