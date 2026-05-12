# src/plots.py
import numpy as np
import matplotlib.pyplot as plt


def _velocity_reference(data):
    v_des = np.asarray(data['v_des'], dtype=float)
    n = len(data['t'])
    if v_des.ndim == 1:
        return np.tile(v_des.reshape(1, 2), (n, 1))
    return v_des


def _position_error(data):
    target = data.get('target_pos')
    if target is None:
        return None
    target = np.asarray(target, dtype=float).reshape(1, 2)
    return np.linalg.norm(data['states'][:, 0:2] - target, axis=1)


def _overshoot_along_path(data):
    target = data.get('target_pos')
    if target is None:
        return None
    p = data['states'][:, 0:2]
    p0 = p[0]
    target = np.asarray(target, dtype=float).reshape(2)
    path = target - p0
    path_len = float(np.linalg.norm(path))
    if path_len < 1e-9:
        return np.zeros(len(p), dtype=float)
    direction = path / path_len
    progress = (p - p0.reshape(1, 2)) @ direction
    return np.maximum(progress - path_len, 0.0)


def plot_results(data, save_path=None, show=True):
    """Single-run 1D height plots."""
    t = data['t']
    s = data['states']
    u = data['controls']
    w = data['winds']
    target = data['target']

    z = s[:, 0]
    v = s[:, 1]
    T = s[:, 2]
    error = np.abs(z - target)

    fig, axes = plt.subplots(3, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(t, z, label='z')
    ax.axhline(target, color='r', linestyle='--', label=f'target = {target:.2f}')
    ax.set_title('Height z [m]')
    ax.legend()
    ax.grid(True)

    ax = axes[0, 1]
    ax.plot(t, v, label='v')
    ax.set_title('Velocity [m/s]')
    ax.legend()
    ax.grid(True)

    ax = axes[1, 0]
    ax.plot(t, T, label='thrust T')
    ax.set_title('Motor thrust [N]')
    ax.legend()
    ax.grid(True)

    ax = axes[1, 1]
    ax.plot(t, u, label='command u')
    ax.set_title('Control [N]')
    ax.legend()
    ax.grid(True)

    ax = axes[2, 0]
    ax.plot(t, w, label='wind force')
    ax.set_title('Disturbance force [N]')
    ax.legend()
    ax.grid(True)

    ax = axes[2, 1]
    ax.plot(t, error, 'k', label='|z - z_des|')
    ax.set_title('Height error [m]')
    ax.legend()
    ax.grid(True)

    fig.suptitle(f'Final error = {error[-1]:.4f} m, t_end = {t[-1]:.2f} s')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_compare(data_pid, data_bs, label_pid='PID', label_bs='Backstepping',
                 save_path=None, show=True):
    """Compare two 1D controllers."""
    t_pid = data_pid['t']
    z_pid = data_pid['states'][:, 0]
    t_bs = data_bs['t']
    z_bs = data_bs['states'][:, 0]
    target = data_pid['target']

    err_pid = np.abs(z_pid - target)
    err_bs = np.abs(z_bs - target)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    ax = axes[0, 0]
    ax.plot(t_pid, z_pid, label=label_pid)
    ax.plot(t_bs, z_bs, label=label_bs)
    ax.axhline(target, color='k', linestyle=':', label='target')
    ax.set_title('Height')
    ax.legend()
    ax.grid(True)

    ax = axes[0, 1]
    ax.plot(t_pid, err_pid, label=label_pid)
    ax.plot(t_bs, err_bs, label=label_bs)
    ax.set_title('Height error [m]')
    ax.legend()
    ax.grid(True)

    ax = axes[1, 0]
    ax.plot(data_pid['t'], data_pid['controls'], label=label_pid)
    ax.plot(data_bs['t'], data_bs['controls'], label=label_bs)
    ax.set_title('Command u [N]')
    ax.legend()
    ax.grid(True)

    ax = axes[1, 1]
    ax.axis('off')
    iae_pid = np.trapezoid(err_pid, t_pid)
    iae_bs = np.trapezoid(err_bs, t_bs)
    ise_pid = np.trapezoid(err_pid ** 2, t_pid)
    ise_bs = np.trapezoid(err_bs ** 2, t_bs)
    rows = [
        ('IAE', f'{iae_pid:.4f}', f'{iae_bs:.4f}'),
        ('ISE', f'{ise_pid:.4f}', f'{ise_bs:.4f}'),
        ('final err', f'{err_pid[-1]:.4f}', f'{err_bs[-1]:.4f}'),
        ('max err', f'{err_pid.max():.4f}', f'{err_bs.max():.4f}'),
        ('t_end', f'{t_pid[-1]:.2f}', f'{t_bs[-1]:.2f}'),
    ]
    table = ax.table(cellText=rows, colLabels=('metric', label_pid, label_bs),
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)
    ax.set_title('Metrics')

    fig.suptitle(f'{label_pid} vs {label_bs}')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_results_2d(data, save_path=None, show=True):
    """2D run: trajectory, velocities, currents, pitch, torque, tracking error."""
    t = data['t']
    s = data['states']
    u = data['controls']
    v_ref = _velocity_reference(data)
    target_pos = data.get('target_pos')

    x, z = s[:, 0], s[:, 1]
    vx, vz = s[:, 2], s[:, 3]
    th = s[:, 4]
    tau = data['taus']

    fig, axes = plt.subplots(3, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.plot(x, z, 'b-', label='path')
    ax.scatter([x[0]], [z[0]], c='g', s=40, zorder=5, label='start')
    ax.scatter([x[-1]], [z[-1]], c='r', s=40, zorder=5, label='end')
    if target_pos is not None:
        ax.scatter([target_pos[0]], [target_pos[1]], marker='*', c='gold',
                   edgecolors='k', s=130, zorder=6, label='target')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('z [m]')
    ax.set_title('x–z plane')
    ax.legend()
    ax.grid(True)

    ax = axes[0, 1]
    ax.plot(t, vx, label='vx')
    ax.plot(t, vz, label='vz')
    ax.plot(t, v_ref[:, 0], color='k', linestyle='--', alpha=0.6, label='vx ref')
    ax.plot(t, v_ref[:, 1], color='k', linestyle=':', alpha=0.6, label='vz ref')
    ax.set_title('Velocities [m/s]')
    ax.legend()
    ax.grid(True)

    ax = axes[1, 0]
    ax.plot(t, u[:, 0], label='I_L')
    ax.plot(t, u[:, 1], label='I_R')
    ax.set_title('Motor current commands [A]')
    ax.legend()
    ax.grid(True)

    ax = axes[1, 1]
    ax.plot(t, np.rad2deg(th), label='theta')
    ax.set_title('Pitch [deg]')
    ax.legend()
    ax.grid(True)

    ax = axes[2, 0]
    ax.plot(t, tau, 'm-', label='tau')
    ax.set_title('Pitch torque [N·m]')
    ax.legend()
    ax.grid(True)

    ax = axes[2, 1]
    ep = _position_error(data)
    if ep is None:
        ev = np.linalg.norm(s[:, 2:4] - v_ref, axis=1)
        ax.plot(t, ev, 'k', label='||v - v_ref||')
        ax.set_title('Speed error [m/s]')
    else:
        ax.plot(t, ep, 'k', label='||p - p*||')
        ax.set_title('Position error [m]')
    ax.legend()
    ax.grid(True)

    fig.suptitle(f'2D simulation, t_end = {t[-1]:.2f} s')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_compare_2d(data_a, data_b, label_a='P', label_b='Backstepping',
                    save_path=None, show=True):
    """Compare two 2D runs (legacy two-way)."""
    t1, t2 = data_a['t'], data_b['t']
    ep1 = _position_error(data_a)
    ep2 = _position_error(data_b)
    if ep1 is None:
        ep1 = np.linalg.norm(data_a['states'][:, 2:4] - _velocity_reference(data_a), axis=1)
        ep2 = np.linalg.norm(data_b['states'][:, 2:4] - _velocity_reference(data_b), axis=1)
        err_title = 'Speed error ||v - v_ref||'
    else:
        err_title = 'Position error ||p - p*||'

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(data_a['states'][:, 0], data_a['states'][:, 1], label=label_a, alpha=0.8)
    ax.plot(data_b['states'][:, 0], data_b['states'][:, 1], label=label_b, alpha=0.8)
    target = data_b.get('target_pos')
    if target is not None:
        ax.scatter([target[0]], [target[1]], marker='*', c='gold',
                   edgecolors='k', s=130, zorder=6, label='target')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('z [m]')
    ax.set_title('Trajectories')
    ax.legend()
    ax.grid(True)

    ax = axes[0, 1]
    ax.plot(t1, ep1, label=label_a)
    ax.plot(t2, ep2, label=label_b)
    ax.set_title(err_title)
    ax.set_xlabel('t [s]')
    ax.legend()
    ax.grid(True)

    ax = axes[1, 0]
    ax.plot(t1, data_a['controls'][:, 0], linestyle='--', alpha=0.7)
    ax.plot(t1, data_a['controls'][:, 1], linestyle='--', alpha=0.7, label=f'{label_a} I')
    ax.plot(t2, data_b['controls'][:, 0], alpha=0.9)
    ax.plot(t2, data_b['controls'][:, 1], alpha=0.9, label=f'{label_b} I')
    ax.set_title('Currents [A]')
    ax.legend()
    ax.grid(True)

    ax = axes[1, 1]
    ax.axis('off')
    iae1 = np.trapezoid(ep1, t1)
    iae2 = np.trapezoid(ep2, t2)
    rows = [('IAE', f'{iae1:.4f}', f'{iae2:.4f}'),
            ('final error', f'{ep1[-1]:.4f}', f'{ep2[-1]:.4f}'),
            ('t_end', f'{t1[-1]:.2f}', f'{t2[-1]:.2f}')]
    ax.table(cellText=rows, colLabels=('metric', label_a, label_b),
             loc='center', cellLoc='center')

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_velocity_controllers_comparison(datasets, labels, save_path=None, show=True):
    """
    Compare multiple 2D controllers (P, PD, PID, Backstepping, ...).

    datasets: list of dicts from run_simulation_2d.
    labels: list of str, same length.
    """
    if len(datasets) != len(labels):
        raise ValueError('datasets and labels must have the same length')

    colors = plt.cm.tab10(np.linspace(0, 0.9, len(datasets)))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    ax = axes[0, 0]
    for d, lab, c in zip(datasets, labels, colors):
        ep = _position_error(d)
        if ep is None:
            ep = np.linalg.norm(d['states'][:, 2:4] - _velocity_reference(d), axis=1)
            ylabel = 'Speed error ||v - v_ref|| [m/s]'
        else:
            ylabel = 'Position error ||p - p*|| [m]'
        ax.plot(d['t'], ep, label=lab, color=c, linewidth=1.6)
    ax.set_title(ylabel)
    ax.set_xlabel('t [s]')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True)

    ax = axes[0, 1]
    for d, lab, c in zip(datasets, labels, colors):
        i_mean = 0.5 * (d['controls'][:, 0] + d['controls'][:, 1])
        ax.plot(d['t'], i_mean, color=c, linewidth=1.5, label=lab)
    ax.set_title('Mean motor current (I_L + I_R) / 2 [A]')
    ax.set_xlabel('t [s]')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True)

    ax = axes[1, 0]
    for d, lab, c in zip(datasets, labels, colors):
        ax.plot(d['states'][:, 0], d['states'][:, 1], label=lab, color=c, linewidth=1.4)
    target = datasets[-1].get('target_pos')
    if target is not None:
        ax.scatter([target[0]], [target[1]], marker='*', c='gold',
                   edgecolors='k', s=140, zorder=6, label='target')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('z [m]')
    ax.set_title('x–z trajectories')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True)

    ax = axes[1, 1]
    ax.axis('off')
    rows = []
    for d, lab in zip(datasets, labels):
        ep = _position_error(d)
        if ep is None:
            ep = np.linalg.norm(d['states'][:, 2:4] - _velocity_reference(d), axis=1)
        iae = np.trapezoid(ep, d['t'])
        rows.append([lab, f'{iae:.4f}', f'{ep[-1]:.4f}', f'{d["t"][-1]:.2f}'])
    tbl = ax.table(
        cellText=rows,
        colLabels=('Controller', 'IAE', 'final error', 't_end [s]'),
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.05, 1.45)
    ax.set_title('Performance summary')

    fig.suptitle('Target-point tracking: controller comparison')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_overshoot_comparison(datasets, labels, save_path=None, show=True):
    """Plot target-point overshoot for several controllers."""
    if len(datasets) != len(labels):
        raise ValueError('datasets and labels must have the same length')

    colors = plt.cm.tab10(np.linspace(0, 0.9, len(datasets)))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    rows = []
    for d, lab, c in zip(datasets, labels, colors):
        overshoot = _overshoot_along_path(d)
        if overshoot is None:
            raise ValueError('overshoot plot requires target_pos in every dataset')
        ax.plot(d['t'], overshoot, label=lab, color=c, linewidth=1.6)
        ep = _position_error(d)
        rows.append([
            lab,
            f'{float(np.max(overshoot)):.4f}',
            f'{float(ep[-1]):.4f}',
            f'{float(np.min(ep)):.4f}',
        ])
    ax.set_title('Overshoot beyond target along start-target line')
    ax.set_xlabel('t [s]')
    ax.set_ylabel('overshoot [m]')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True)

    ax = axes[1]
    ax.axis('off')
    tbl = ax.table(
        cellText=rows,
        colLabels=('Controller', 'peak overshoot [m]', 'final ||p-p*|| [m]', 'min ||p-p*|| [m]'),
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.45)
    ax.set_title('Overshoot summary')

    fig.suptitle('Target-point overshoot comparison')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_lyapunov_certificate(data, save_path=None, show=True):
    """
    Lyapunov stability certificate for Backstepping run: composite V(t), terms, numeric dV/dt.

    Expects keys 't', 'lyapunov', 'lyapunov_terms' from run_simulation_2d.
    """
    t = np.asarray(data['t'], dtype=float)
    V = np.asarray(data['lyapunov'], dtype=float)
    terms = data.get('lyapunov_terms', {})

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    ax = axes[0]
    ax.plot(t, V, 'b-', linewidth=1.8, label='V (total)')
    ax.set_ylabel('V')
    ax.set_title('Lyapunov candidate V(t) (sum of weighted squared tracking errors)')
    ax.grid(True)
    ax.legend(loc='upper right')

    ax = axes[1]
    if terms:
        for name, arr in terms.items():
            ax.plot(t, np.asarray(arr, dtype=float), label=name, linewidth=1.2)
    ax.set_ylabel('term value')
    ax.set_title('Components of V')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True)

    ax = axes[2]
    if len(t) > 2 and np.all(np.isfinite(V)):
        Vdot = np.gradient(V, t, edge_order=2)
        ax.plot(t, Vdot, color='darkred', linewidth=1.4, label='dV/dt (numerical)')
        ax.axhline(0.0, color='k', linestyle=':', linewidth=0.8)
    ax.set_xlabel('t [s]')
    ax.set_ylabel('dV/dt')
    ax.set_title('Time derivative of V (numerical; negative values support stability intuition)')
    ax.legend(loc='upper right')
    ax.grid(True)

    fig.suptitle('Stability certificate sketch — Backstepping controller')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    else:
        plt.close(fig)
