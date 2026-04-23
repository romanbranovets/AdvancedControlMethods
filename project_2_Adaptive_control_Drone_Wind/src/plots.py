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
