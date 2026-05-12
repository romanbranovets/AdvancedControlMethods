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

    # Таблица метрик
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