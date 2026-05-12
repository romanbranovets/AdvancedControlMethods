# src/visualization.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def visualize_1d_compare(data_main, data_baseline,
                         label_main='Backstepping', label_baseline='PID',
                         target_fps=20, show=True,
                         save_path=None, save_dpi=120):
    """
    Анимация: вертикальная линия, на которой движутся две точки (два контроллера).
    Дополнительно строятся графики ошибки и управления.
    Исправлено: blit=False, чтобы избежать ошибки с fig.text.
    """
    # Выравниваем длину данных
    n_max = max(len(data_main['t']), len(data_baseline['t']))

    def pad(data, n):
        if len(data['t']) >= n:
            return data
        pad_len = n - len(data['t'])
        dt = np.median(np.diff(data['t'])) if len(data['t']) > 1 else 0.005
        t_extra = data['t'][-1] + dt * np.arange(1, pad_len + 1)
        return {
            't': np.concatenate([data['t'], t_extra]),
            'states': np.concatenate([data['states'],
                                      np.tile(data['states'][-1], (pad_len, 1))]),
            'controls': np.concatenate([data['controls'],
                                        np.full(pad_len, data['controls'][-1])]),
            'target': data['target'],
        }

    data_main = pad(data_main, n_max)
    data_baseline = pad(data_baseline, n_max)
    t = data_main['t']
    target = data_main['target']

    # Субдискретизация под fps
    dt_sim = np.median(np.diff(t)) if len(t) > 1 else 0.005
    stride = max(1, int(round(1.0 / (target_fps * dt_sim))))
    idx = np.arange(0, len(t), stride)
    if idx[-1] != len(t) - 1:
        idx = np.append(idx, len(t) - 1)

    fig, (ax_anim, ax_err) = plt.subplots(1, 2, figsize=(10, 5),
                                          gridspec_kw={'width_ratios': [1, 2]})

    # Левая панель – вертикальная шкала
    ax_anim.set_xlim(-0.5, 0.5)
    z_all = np.concatenate([data_main['states'][:, 0], data_baseline['states'][:, 0], [target]])
    z_min, z_max = z_all.min() - 1, z_all.max() + 1
    ax_anim.set_ylim(z_min, z_max)
    ax_anim.axhline(target, color='red', linestyle='--', alpha=0.7, label='Target')
    ax_anim.set_ylabel('Height z [m]')
    ax_anim.set_xticks([])
    ax_anim.set_title('Drone position')

    point_main, = ax_anim.plot([], [], 'bo', markersize=10, label=label_main)
    point_base, = ax_anim.plot([], [], 's', color='gray', markersize=10,
                               alpha=0.6, label=label_baseline)
    ax_anim.legend(loc='upper right')

    # Правая панель – ошибка от времени
    err_main = np.abs(data_main['states'][:, 0] - target)
    err_base = np.abs(data_baseline['states'][:, 0] - target)
    ax_err.plot(t, err_main, 'b-', label=label_main)
    ax_err.plot(t, err_base, 'gray', linestyle='--', label=label_baseline)
    ax_err.set_xlim(0, t[-1])
    ax_err.set_ylim(0, max(err_main.max(), err_base.max()) * 1.1)
    ax_err.set_xlabel('t [s]')
    ax_err.set_ylabel('|error| [m]')
    ax_err.legend()
    ax_err.grid(True)
    cursor, = ax_err.plot([], [], 'ko', markersize=6)

    time_text = fig.text(0.5, 0.02, '', ha='center', fontsize=10)

    def update(frame):
        f = idx[frame]
        z_main = data_main['states'][f, 0]
        z_base = data_baseline['states'][f, 0]

        point_main.set_data([0], [z_main])
        point_base.set_data([0], [z_base])

        cursor.set_data([t[f]], [err_main[f]])

        time_text.set_text(f't = {t[f]:.2f} s')
        return point_main, point_base, cursor

    ani = FuncAnimation(fig, update, frames=len(idx),
                        interval=1000 / target_fps, blit=False, repeat=True)
    fig._ani_ref = ani

    if save_path:
        try:
            ani.save(save_path, writer='pillow', fps=target_fps, dpi=save_dpi)
            print(f"[viz] saved {save_path}")
        except Exception as e:
            print(f"[viz] save failed: {e}")

    if show:
        plt.show()
    return ani


def visualize_2d_drone(data, L_body=0.36, target_fps=25, show=True,
                       save_path=None, save_dpi=120):
    """
    2D x–z animation: drone body as a line segment oriented by pitch theta.
    data: output of run_simulation_2d.
    """
    t = data['t']
    s = data['states']
    v_des = np.asarray(data['v_des'], dtype=float)
    if v_des.ndim == 1:
        v_des = np.tile(v_des.reshape(1, 2), (len(t), 1))
    target_pos = data.get('target_pos')


    dt_sim = np.median(np.diff(t)) if len(t) > 1 else 0.005
    stride = max(1, int(round(1.0 / (target_fps * dt_sim))))
    idx = np.arange(0, len(t), stride)
    if idx[-1] != len(t) - 1:
        idx = np.append(idx, len(t) - 1)

    fig, (ax_world, ax_ev) = plt.subplots(1, 2, figsize=(11, 5),
                                          gridspec_kw={'width_ratios': [1.2, 1]})

    pad = 1.0
    x_all, z_all = s[:, 0], s[:, 1]
    ax_world.set_xlim(x_all.min() - pad, x_all.max() + pad)
    ax_world.set_ylim(z_all.min() - pad, z_all.max() + pad)
    ax_world.set_aspect('equal', adjustable='box')
    ax_world.set_xlabel('x [m]')
    ax_world.set_ylabel('z [m]')
    ax_world.set_title('Motion in x–z')
    ax_world.grid(True, alpha=0.3)
    if target_pos is not None:
        ax_world.scatter([target_pos[0]], [target_pos[1]], marker='*', c='gold',
                         edgecolors='k', s=140, zorder=6, label='target')
        ax_world.legend(loc='best')

    traj, = ax_world.plot([], [], 'b-', alpha=0.35, linewidth=1)
    body_line, = ax_world.plot([], [], '-', color='0.2', lw=3, solid_capstyle='round')

    if target_pos is None:
        err = np.linalg.norm(s[:, 2:4] - v_des, axis=1)
        err_ylabel = '||v - v_ref|| [m/s]'
        err_title = 'Speed error'
    else:
        err = np.linalg.norm(s[:, 0:2] - np.asarray(target_pos).reshape(1, 2), axis=1)
        err_ylabel = '||p - p*|| [m]'
        err_title = 'Position error'
    ax_ev.plot(t, err, 'k-', alpha=0.5)
    cursor_ev, = ax_ev.plot([], [], 'ro', markersize=5)
    ax_ev.set_xlim(0, t[-1])
    ax_ev.set_ylim(0, max(err.max(), 1e-3) * 1.1)
    ax_ev.set_xlabel('t [s]')
    ax_ev.set_ylabel(err_ylabel)
    ax_ev.set_title(err_title)
    ax_ev.grid(True)

    time_text = fig.text(0.5, 0.02, '', ha='center', fontsize=10)

    def update(frame):
        f = idx[frame]
        x, z, _, _, theta = s[f, 0], s[f, 1], s[f, 2], s[f, 3], s[f, 4]
        traj.set_data(s[: f + 1, 0], s[: f + 1, 1])
        ct, st = np.cos(theta), np.sin(theta)
        hx = 0.5 * L_body * ct
        hz = 0.5 * L_body * st
        body_line.set_data([x - hx, x + hx], [z - hz, z + hz])
        cursor_ev.set_data([t[f]], [err[f]])
        if target_pos is None:
            time_text.set_text(f't = {t[f]:.2f} s, vx_ref={v_des[f, 0]:.2f}, vz_ref={v_des[f, 1]:.2f}')
        else:
            time_text.set_text(f't = {t[f]:.2f} s, target=({target_pos[0]:.2f}, {target_pos[1]:.2f})')
        return traj, body_line, cursor_ev

    ani = FuncAnimation(fig, update, frames=len(idx),
                        interval=1000 / target_fps, blit=False, repeat=True)
    fig._ani_ref = ani

    if save_path:
        try:
            ani.save(save_path, writer='pillow', fps=target_fps, dpi=save_dpi)
            print(f"[viz] saved {save_path}")
        except Exception as e:
            print(f"[viz] save failed: {e}")

    if show:
        plt.show()
    return ani
