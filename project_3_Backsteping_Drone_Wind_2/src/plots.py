# src/plots.py
import numpy as np
import matplotlib.pyplot as plt


def plot_results(data, save_path=None, show=True):
    """Графики для одной симуляции (высота, скорость, тяга, управление, ветер)."""
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
    ax.axhline(target, color='r', linestyle='--', label=f'target={target:.2f}')
    ax.set_title('Высота [м]')
    ax.legend(); ax.grid(True)

    ax = axes[0, 1]
    ax.plot(t, v, label='v')
    ax.set_title('Скорость [м/с]')
    ax.legend(); ax.grid(True)

    ax = axes[1, 0]
    ax.plot(t, T, label='тяга T')
    ax.set_title('Тяга мотора [Н]')
    ax.legend(); ax.grid(True)

    ax = axes[1, 1]
    ax.plot(t, u, label='команда u')
    ax.set_title('Управляющий сигнал [Н]')
    ax.legend(); ax.grid(True)

    ax = axes[2, 0]
    ax.plot(t, w, label='сила ветра')
    ax.set_title('Внешняя сила [Н]')
    ax.legend(); ax.grid(True)

    ax = axes[2, 1]
    ax.plot(t, error, 'k', label='|z - z_des|')
    ax.set_title('Ошибка высоты [м]')
    ax.legend(); ax.grid(True)

    fig.suptitle(f'Финальная ошибка = {error[-1]:.4f} м, время = {t[-1]:.2f} с')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_compare(data_pid, data_bs, label_pid='PID', label_bs='Backstepping',
                 save_path=None, show=True):
    """Сравнение двух контроллеров: высота, ошибка, управление."""
    t_pid = data_pid['t']; z_pid = data_pid['states'][:, 0]
    t_bs  = data_bs['t'];  z_bs  = data_bs['states'][:, 0]
    target = data_pid['target']

    err_pid = np.abs(z_pid - target)
    err_bs  = np.abs(z_bs - target)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    ax = axes[0, 0]
    ax.plot(t_pid, z_pid, label=label_pid)
    ax.plot(t_bs,  z_bs,  label=label_bs)
    ax.axhline(target, color='k', linestyle=':', label='target')
    ax.set_title('Высота')
    ax.legend(); ax.grid(True)

    ax = axes[0, 1]
    ax.plot(t_pid, err_pid, label=label_pid)
    ax.plot(t_bs,  err_bs,  label=label_bs)
    ax.set_title('Ошибка высоты [м]')
    ax.legend(); ax.grid(True)

    ax = axes[1, 0]
    ax.plot(data_pid['t'], data_pid['controls'], label=label_pid)
    ax.plot(data_bs['t'],  data_bs['controls'],  label=label_bs)
    ax.set_title('Команда u [Н]')
    ax.legend(); ax.grid(True)

    ax = axes[1, 1]
    ax.axis('off')
    iae_pid = np.trapezoid(err_pid, t_pid)
    iae_bs  = np.trapezoid(err_bs,  t_bs)
    ise_pid = np.trapezoid(err_pid**2, t_pid)
    ise_bs  = np.trapezoid(err_bs**2,  t_bs)
    rows = [
        ('IAE',        f'{iae_pid:.4f}', f'{iae_bs:.4f}'),
        ('ISE',        f'{ise_pid:.4f}', f'{ise_bs:.4f}'),
        ('final err',  f'{err_pid[-1]:.4f}', f'{err_bs[-1]:.4f}'),
        ('max err',    f'{err_pid.max():.4f}', f'{err_bs.max():.4f}'),
        ('t_end',      f'{t_pid[-1]:.2f}',     f'{t_bs[-1]:.2f}'),
    ]
    table = ax.table(cellText=rows, colLabels=('metric', label_pid, label_bs),
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)
    ax.set_title('Сравнение метрик')

    fig.suptitle(f'{label_pid} vs {label_bs}')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_results_2d(data, save_path=None, show=True):
    """Графики 2D: траектория x-z, скорости, токи, угол, момент."""
    t = data['t']
    s = data['states']
    u = data['controls']
    v_des = data['v_des']

    x, z = s[:, 0], s[:, 1]
    vx, vz = s[:, 2], s[:, 3]
    th = s[:, 4]
    tau = data['taus']

    fig, axes = plt.subplots(3, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.plot(x, z, 'b-', label='траектория')
    ax.scatter([x[0]], [z[0]], c='g', s=40, zorder=5, label='старт')
    ax.scatter([x[-1]], [z[-1]], c='r', s=40, zorder=5, label='финиш')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x [м]')
    ax.set_ylabel('z [м]')
    ax.set_title('Плоскость x–z')
    ax.legend(); ax.grid(True)

    ax = axes[0, 1]
    ax.plot(t, vx, label='vx')
    ax.plot(t, vz, label='vz')
    ax.axhline(v_des[0], color='k', linestyle='--', alpha=0.6, label=f'vx*={v_des[0]:.2f}')
    ax.axhline(v_des[1], color='k', linestyle=':', alpha=0.6, label=f'vz*={v_des[1]:.2f}')
    ax.set_title('Скорости [м/с]')
    ax.legend(); ax.grid(True)

    ax = axes[1, 0]
    ax.plot(t, u[:, 0], label='I_L')
    ax.plot(t, u[:, 1], label='I_R')
    ax.set_title('Команды тока [А]')
    ax.legend(); ax.grid(True)

    ax = axes[1, 1]
    ax.plot(t, np.rad2deg(th), label='θ')
    ax.set_title('Тангаж [град]')
    ax.legend(); ax.grid(True)

    ax = axes[2, 0]
    ax.plot(t, tau, 'm-', label='τ')
    ax.set_title('Момент сил [Н·м]')
    ax.legend(); ax.grid(True)

    ax = axes[2, 1]
    ev = np.linalg.norm(s[:, 2:4] - v_des.reshape(1, 2), axis=1)
    ax.plot(t, ev, 'k', label='||v - v*||')
    ax.set_title('Ошибка скорости [м/с]')
    ax.legend(); ax.grid(True)

    fig.suptitle(f'2D симуляция, t_end = {t[-1]:.2f} с')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_compare_2d(data_pd, data_bs, label_pd='PD+инверсия', label_bs='Backstepping',
                    save_path=None, show=True):
    t1, t2 = data_pd['t'], data_bs['t']
    ev1 = np.linalg.norm(data_pd['states'][:, 2:4] - data_pd['v_des'].reshape(1, 2), axis=1)
    ev2 = np.linalg.norm(data_bs['states'][:, 2:4] - data_bs['v_des'].reshape(1, 2), axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(data_pd['states'][:, 0], data_pd['states'][:, 1], label=label_pd, alpha=0.8)
    ax.plot(data_bs['states'][:, 0], data_bs['states'][:, 1], label=label_bs, alpha=0.8)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x [м]'); ax.set_ylabel('z [м]')
    ax.set_title('Траектории')
    ax.legend(); ax.grid(True)

    ax = axes[0, 1]
    ax.plot(t1, ev1, label=label_pd)
    ax.plot(t2, ev2, label=label_bs)
    ax.set_title('Ошибка скорости')
    ax.legend(); ax.grid(True)

    ax = axes[1, 0]
    ax.plot(t1, data_pd['controls'][:, 0], linestyle='--', alpha=0.7)
    ax.plot(t1, data_pd['controls'][:, 1], linestyle='--', alpha=0.7, label=f'{label_pd} I')
    ax.plot(t2, data_bs['controls'][:, 0], alpha=0.9)
    ax.plot(t2, data_bs['controls'][:, 1], alpha=0.9, label=f'{label_bs} I')
    ax.set_title('Токи [А]')
    ax.legend(); ax.grid(True)

    ax = axes[1, 1]
    ax.axis('off')
    iae1 = np.trapezoid(ev1, t1)
    iae2 = np.trapezoid(ev2, t2)
    rows = [('IAE ||e_v||', f'{iae1:.4f}', f'{iae2:.4f}'),
            ('final ||e_v||', f'{ev1[-1]:.4f}', f'{ev2[-1]:.4f}'),
            ('t_end', f'{t1[-1]:.2f}', f'{t2[-1]:.2f}')]
    ax.table(cellText=rows, colLabels=('metric', label_pd, label_bs),
             loc='center', cellLoc='center')

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    else:
        plt.close(fig)
