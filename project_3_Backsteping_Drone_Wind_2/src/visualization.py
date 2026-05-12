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
    ax_anim.set_ylabel('Высота z [м]')
    ax_anim.set_xticks([])
    ax_anim.set_title('Положение дрона')

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
    ax_err.set_xlabel('t [с]')
    ax_err.set_ylabel('|ошибка| [м]')
    ax_err.legend()
    ax_err.grid(True)
    cursor, = ax_err.plot([], [], 'ko', markersize=6)

    # Время – выводим через fig.text
    time_text = fig.text(0.5, 0.02, '', ha='center', fontsize=10)

    def update(frame):
        f = idx[frame]
        z_main = data_main['states'][f, 0]
        z_base = data_baseline['states'][f, 0]

        point_main.set_data([0], [z_main])
        point_base.set_data([0], [z_base])

        cursor.set_data([t[f]], [err_main[f]])

        time_text.set_text(f't = {t[f]:.2f} с')
        # Возвращаем только axes-художников, time_text исключаем
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