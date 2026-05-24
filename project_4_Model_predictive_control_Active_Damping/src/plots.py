# src/plots.py
import numpy as np
import matplotlib.pyplot as plt

def plot_results(data, save_path=None, show=True):
    t = data['t']
    states = data['states']   # x1, x2, v1, v2
    controls = data['controls']
    dist = data['disturbances']
    ref = data['ref']

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Позиции
    ax = axes[0, 0]
    ax.plot(t, states[:, 0], label='x1 (main mass)')
    ax.plot(t, states[:, 1], label='x2 (absorber)')
    ax.axhline(ref[0], color='r', linestyle='--', label='x1_ref')
    ax.axhline(ref[1], color='g', linestyle='--', label='x2_ref')
    ax.set_ylabel('Position [m]')
    ax.legend()
    ax.grid(True)

    # Скорости
    ax = axes[0, 1]
    ax.plot(t, states[:, 2], label='v1')
    ax.plot(t, states[:, 3], label='v2')
    ax.set_ylabel('Velocity [m/s]')
    ax.legend()
    ax.grid(True)

    # Управление
    ax = axes[1, 0]
    ax.plot(t, controls, label='Actuator force')
    ax.set_ylabel('Control force [N]')
    ax.legend()
    ax.grid(True)

    # Возмущение
    ax = axes[1, 1]
    ax.plot(t, dist, label='disturbance')
    ax.set_ylabel('Disturbance [N]')
    ax.legend()
    ax.grid(True)

    fig.suptitle(f'Vibration damping (two‑mass system, t_end = {t[-1]:.2f} s)')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_comparison(data_mpc, data_pid, label_mpc='MPC', label_pid='PID',
                    save_path=None, show=True):
    t1, t2 = data_mpc['t'], data_pid['t']
    # Ошибка только по основной массе
    err1 = np.abs(data_mpc['states'][:, 0] - data_mpc['ref'][0])
    err2 = np.abs(data_pid['states'][:, 0] - data_pid['ref'][0])

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    ax = axes[0, 0]
    ax.plot(t1, data_mpc['states'][:, 0], label='x1 (MPC)')
    ax.plot(t2, data_pid['states'][:, 0], '--', label='x1 (PID)')
    ax.axhline(data_mpc['ref'][0], color='k', linestyle=':', label='ref')
    ax.set_ylabel('x1 [m]')
    ax.legend()
    ax.grid(True)

    ax = axes[0, 1]
    ax.plot(t1, data_mpc['states'][:, 1], label='x2 (MPC)')
    ax.plot(t2, data_pid['states'][:, 1], '--', label='x2 (PID)')
    ax.axhline(data_mpc['ref'][1], color='k', linestyle=':', label='ref')
    ax.set_ylabel('x2 [m]')
    ax.legend()
    ax.grid(True)

    ax = axes[1, 0]
    ax.plot(t1, err1, label=label_mpc)
    ax.plot(t2, err2, label=label_pid)
    ax.set_ylabel('|x1 error| [m]')
    ax.set_xlabel('t [s]')
    ax.legend()
    ax.grid(True)

    ax = axes[1, 1]
    iae_mpc = np.trapezoid(err1, t1)
    iae_pid = np.trapezoid(err2, t2)
    rows = [('IAE', f'{iae_mpc:.4f}', f'{iae_pid:.4f}'),
            ('final error', f'{err1[-1]:.4f}', f'{err2[-1]:.4f}'),
            ('max error', f'{err1.max():.4f}', f'{err2.max():.4f}')]
    ax.axis('off')
    table = ax.table(cellText=rows, colLabels=('Metric', label_mpc, label_pid),
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    ax.set_title('Performance metrics (main mass)')

    fig.suptitle('MPC vs PID for two‑mass vibration absorber')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_trajectory_2d(data_mpc, data_pid, save_path=None, show=True):
    """Для одномерной системы траектория не имеет смысла – выводим фазовую плоскость x1/v1."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(data_mpc['states'][:, 0], data_mpc['states'][:, 2], 'b-', label='MPC')
    ax.plot(data_pid['states'][:, 0], data_pid['states'][:, 2], 'r--', label='PID')
    ax.set_xlabel('x1 [m]')
    ax.set_ylabel('v1 [m/s]')
    ax.set_aspect('auto')
    ax.legend()
    ax.grid(True)
    ax.set_title('Phase portrait (main mass)')
    if save_path:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    else:
        plt.close(fig)