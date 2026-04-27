# src/plots.py
import numpy as np
import matplotlib.pyplot as plt


def plot_results(data, save_path=None, show=True):
    t = data['t']
    s = data['states']
    u = data['controls']
    w = data['winds']
    target = data['target']

    err = np.linalg.norm(s[:, 0:3] - target, axis=1)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # position
    ax = axes[0, 0]
    for i, name in enumerate(['x', 'y', 'z']):
        ax.plot(t, s[:, i], label=name)
        ax.axhline(target[i], linestyle='--', color=f'C{i}', alpha=0.4)
    ax.set_title('Position [m]')
    ax.set_xlabel('t [s]')
    ax.legend()
    ax.grid(True)

    # velocity
    ax = axes[0, 1]
    for i, name in enumerate(['vx', 'vy', 'vz']):
        ax.plot(t, s[:, 3 + i], label=name)
    ax.set_title('Velocity [m/s]')
    ax.set_xlabel('t [s]')
    ax.legend()
    ax.grid(True)

    # euler angles
    ax = axes[1, 0]
    for i, name in enumerate([r'$\phi$ (roll)', r'$\theta$ (pitch)', r'$\psi$ (yaw)']):
        ax.plot(t, np.rad2deg(s[:, 6 + i]), label=name)
    ax.set_title('Euler angles [deg]')
    ax.set_xlabel('t [s]')
    ax.legend()
    ax.grid(True)

    # body rates
    ax = axes[1, 1]
    for i, name in enumerate(['p', 'q', 'r']):
        ax.plot(t, s[:, 9 + i], label=name)
    ax.set_title('Body angular rates [rad/s]')
    ax.set_xlabel('t [s]')
    ax.legend()
    ax.grid(True)

    # motor thrusts
    ax = axes[2, 0]
    for i in range(4):
        ax.plot(t, u[:, i], label=f'u{i+1}')
    ax.set_title('Motor thrust [N]')
    ax.set_xlabel('t [s]')
    ax.legend()
    ax.grid(True)

    # wind + tracking error
    ax = axes[2, 1]
    for i, name in enumerate(['wx', 'wy', 'wz']):
        ax.plot(t, w[:, i], label=name)
    ax2 = ax.twinx()
    ax2.plot(t, err, color='k', alpha=0.5, linestyle='--', label='||pos-target||')
    ax2.set_ylabel('tracking error [m]')
    ax.set_title('Wind [m/s] (left) & tracking error (right)')
    ax.set_xlabel('t [s]')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True)

    fig.suptitle(f'final error = {err[-1]:.3f} m  |  max error = {err.max():.2f} m  |  t_final = {t[-1]:.2f} s')
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_compare(data_pid, data_mrac, save_path=None, show=True):
    """Overlay tracking error and key signals from PID vs MRAC runs."""
    target = data_pid['target']
    err_pid  = np.linalg.norm(data_pid['states'][:, 0:3]  - target, axis=1)
    err_mrac = np.linalg.norm(data_mrac['states'][:, 0:3] - target, axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    ax = axes[0, 0]
    ax.plot(data_pid['t'],  err_pid,  label='PID baseline', linewidth=2)
    ax.plot(data_mrac['t'], err_mrac, label='MRAC',         linewidth=2)
    ax.set_title('Tracking error  ||pos - target|| [m]')
    ax.set_xlabel('t [s]')
    ax.legend(); ax.grid(True)

    ax = axes[0, 1]
    for i, name in enumerate([r'$\phi$', r'$\theta$', r'$\psi$']):
        ax.plot(data_pid['t'],  np.rad2deg(data_pid['states'][:, 6 + i]),
                linestyle='--', alpha=0.6, label=f'PID  {name}', color=f'C{i}')
        ax.plot(data_mrac['t'], np.rad2deg(data_mrac['states'][:, 6 + i]),
                linestyle='-', label=f'MRAC {name}', color=f'C{i}')
    ax.set_title('Euler angles [deg]')
    ax.set_xlabel('t [s]')
    ax.legend(ncol=2, fontsize=8); ax.grid(True)

    ax = axes[1, 0]
    for i, name in enumerate(['p', 'q', 'r']):
        ax.plot(data_pid['t'],  data_pid['states'][:, 9 + i],
                linestyle='--', alpha=0.6, label=f'PID  {name}', color=f'C{i}')
        ax.plot(data_mrac['t'], data_mrac['states'][:, 9 + i],
                linestyle='-', label=f'MRAC {name}', color=f'C{i}')
    ax.set_title('Body rates [rad/s]')
    ax.set_xlabel('t [s]')
    ax.legend(ncol=2, fontsize=8); ax.grid(True)

    ax = axes[1, 1]
    iae_pid  = np.trapz(err_pid,  data_pid['t'])
    iae_mrac = np.trapz(err_mrac, data_mrac['t'])
    ise_pid  = np.trapz(err_pid**2,  data_pid['t'])
    ise_mrac = np.trapz(err_mrac**2, data_mrac['t'])
    rows = [
        ('IAE',         f'{iae_pid:.3f}',  f'{iae_mrac:.3f}'),
        ('ISE',         f'{ise_pid:.3f}',  f'{ise_mrac:.3f}'),
        ('final err',   f'{err_pid[-1]:.3f}', f'{err_mrac[-1]:.3f}'),
        ('max  err',    f'{err_pid.max():.3f}', f'{err_mrac.max():.3f}'),
        ('t_end',       f'{data_pid["t"][-1]:.2f}', f'{data_mrac["t"][-1]:.2f}'),
    ]
    ax.axis('off')
    table = ax.table(cellText=[r for r in rows],
                     colLabels=['metric', 'PID', 'MRAC'],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.6)
    ax.set_title('Performance summary')

    fig.suptitle('PID baseline  vs  MRAC + sigma-modification')
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_adaptation(history, save_path=None, show=True):
    """Visualize MRAC adaptation on velocity loop: v tracking + Theta evolution."""
    t = history['t']
    v = history['v']
    v_m = history['v_m']
    v_des = history['v_des']
    theta = history['theta']  # (T, 3, n_basis)

    axis_names = ['vx', 'vy', 'vz']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for j in range(3):
        ax = axes[0, j]
        ax.plot(t, v_des[:, j], 'k:',  label=r'$v_{des}$')
        ax.plot(t, v_m[:, j],   'b--', label=r'$v_m$ (ref. model)')
        ax.plot(t, v[:, j],     'r-',  label=r'$v$ (plant)')
        ax.set_title(f'{axis_names[j]} [m/s]')
        ax.set_xlabel('t [s]')
        ax.legend(fontsize=8); ax.grid(True)

        ax = axes[1, j]
        n_basis = theta.shape[2]
        labels = ([r'$\theta_0$ (bias)',
                   r'$\theta_1$ ($v$)',
                   r'$\theta_2$ ($|v|v$)'])[:n_basis]
        for k in range(n_basis):
            ax.plot(t, theta[:, j, k], label=labels[k])
        ax.set_title(f'{axis_names[j]} adaptive params')
        ax.set_xlabel('t [s]')
        ax.legend(fontsize=8); ax.grid(True)

    fig.suptitle('MRAC adaptation history (velocity loop)')
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    else:
        plt.close(fig)
