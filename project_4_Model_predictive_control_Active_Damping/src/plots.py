# src/plots.py
"""
Графики, ориентированные на доказательство сходимости и честное сравнение.

convergence_proof:
    Главная фигура. Шесть панелей по сценарию свободного отклика (без возмущения):
      (1) Траектории x1(t), x2(t) для MPC/LQR/PID + 2%-полоса.
      (2) Норма состояния ||x(t)|| в log-шкале vs теоретическая огибающая
          C·exp(-α t), где α = -ln(ρ)/dt_control, ρ = max |λ(Ad-BdK)|.
      (3) Функция Ляпунова V(x) = x^T P x в log-шкале — монотонное убывание.
      (4) Прирост Ляпунова ΔV_k = V(x_{k+1}) - V(x_k) ≤ 0 (теоретическое
          условие устойчивости в дискретном времени).
      (5) Управление u(t) с границами насыщения.
      (6) Спектр Ad-BdK на комплексной плоскости с единичной окружностью.

disturbance_rejection:
    Сценарий с гармоническим возмущением — установившаяся амплитуда
    основной массы для трёх контроллеров + open-loop (без управления).

metrics_table:
    Компактная таблица всех метрик.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from .analysis import (
    lqr_gain_and_P, closed_loop_spectrum, lyapunov_series, state_norm,
    settling_time, integral_metrics, control_metrics, fit_exponential_envelope,
    all_metrics,
)


def convergence_proof(data_dict, Ad, Bd, Q, R, save_path=None, show=False,
                      title='Convergence proof: free response (no disturbance)'):
    """data_dict: {label: data}. Должен содержать как минимум LQR-данные для теории."""
    K, P = lqr_gain_and_P(Ad, Bd, Q, R)
    eig, rho = closed_loop_spectrum(Ad, Bd, K)
    # Берём dt_control из первой попавшейся симуляции
    first = next(iter(data_dict.values()))
    dt_c = first.get('dt_control', first['dt'])
    alpha = -np.log(rho) / dt_c if rho > 0 else np.inf

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    colors = {'MPC': 'C0', 'LQR': 'C2', 'PID': 'C3', 'open-loop': 'C7'}
    styles = {'MPC': '-', 'LQR': '--', 'PID': '-.', 'open-loop': ':'}

    # ---- (1) x1, x2 ---------------------------------------------------------
    ax = axes[0, 0]
    for label, d in data_dict.items():
        ax.plot(d['t'], d['states'][:, 0], styles.get(label, '-'),
                color=colors.get(label, None), label=f'x1 ({label})', lw=1.5)
    # 2% полоса от пика MPC
    if 'MPC' in data_dict:
        peak = np.max(np.abs(data_dict['MPC']['states'][:, 0]))
        ax.axhspan(-0.02 * peak, 0.02 * peak, color='gray', alpha=0.15,
                   label='2% band of peak')
    ax.set_xlabel('t [s]')
    ax.set_ylabel('x1 [m]')
    ax.set_title('(1) Main mass position')
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8)

    # ---- (2) ||x|| log-scale + теоретическая экспонента ---------------------
    ax = axes[0, 1]
    for label, d in data_dict.items():
        n = state_norm(d['states'])
        ax.semilogy(d['t'], np.maximum(n, 1e-12), styles.get(label, '-'),
                    color=colors.get(label, None), label=f'||x|| ({label})', lw=1.5)
    # Теоретическая огибающая ||x_k|| <= kappa(V) * rho^k * ||x_0||
    # где V — матрица собственных векторов A_cl, ρ — спектральный радиус.
    if 'LQR' in data_dict:
        d_lqr = data_dict['LQR']
        n0 = np.linalg.norm(d_lqr['states'][0])
        Acl = Ad - Bd @ K
        try:
            eigvals, V = np.linalg.eig(Acl)
            kappa = float(np.linalg.cond(V))
        except np.linalg.LinAlgError:
            kappa = 1.0
        # альфа в "непрерывной" шкале: ρ^k = exp(-α t) при t = k * dt_c
        env = kappa * n0 * np.exp(-alpha * d_lqr['t'])
        ax.semilogy(d_lqr['t'], env, 'k:', lw=2,
                    label=fr'$\kappa(V)\,\|x_0\|\,e^{{-\alpha t}}$ ($\alpha={alpha:.2f}$ s$^{{-1}}$)')
    ax.set_xlabel('t [s]')
    ax.set_ylabel(r'$\|x(t)\|_2$ (log)')
    ax.set_title('(2) State norm vs theoretical envelope')
    ax.grid(True, which='both', alpha=0.4)
    ax.legend(fontsize=8)

    # ---- (3) V(x) Ляпунов ---------------------------------------------------
    ax = axes[0, 2]
    for label, d in data_dict.items():
        V = lyapunov_series(d['states'], P)
        ax.semilogy(d['t'], np.maximum(V, 1e-16), styles.get(label, '-'),
                    color=colors.get(label, None), label=label, lw=1.5)
    ax.set_xlabel('t [s]')
    ax.set_ylabel(r'$V(x) = x^\top P x$ (log)')
    ax.set_title('(3) Lyapunov function (LQR-Riccati P)')
    ax.grid(True, which='both', alpha=0.4)
    ax.legend(fontsize=8)

    # ---- (4) ΔV = V_{k+1} - V_k ≤ 0 -----------------------------------------
    ax = axes[1, 0]
    for label, d in data_dict.items():
        V = lyapunov_series(d['states'], P)
        dV = np.diff(V)
        # сэмплируем разреженно для читаемости
        stride = max(1, len(dV) // 800)
        ax.plot(d['t'][:-1][::stride], dV[::stride], styles.get(label, '-'),
                color=colors.get(label, None), label=label, lw=1.0)
    ax.axhline(0.0, color='k', lw=0.8)
    ax.set_xlabel('t [s]')
    ax.set_ylabel(r'$\Delta V_k = V(x_{k+1}) - V(x_k)$')
    ax.set_title(r'(4) Lyapunov decrement (must be $\leq 0$)')
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8)

    # ---- (5) u(t) -----------------------------------------------------------
    ax = axes[1, 1]
    for label, d in data_dict.items():
        ax.plot(d['t'], d['controls'], styles.get(label, '-'),
                color=colors.get(label, None), label=label, lw=1.2)
    ax.set_xlabel('t [s]')
    ax.set_ylabel('u(t) [N]')
    ax.set_title('(5) Control signal')
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8)

    # ---- (6) спектр Acl -----------------------------------------------------
    ax = axes[1, 2]
    theta = np.linspace(0, 2 * np.pi, 256)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', lw=1)
    ax.scatter(eig.real, eig.imag, s=80, c='C2', marker='o',
               edgecolors='black', zorder=3, label='eig(Ad - Bd K)')
    ax.add_patch(Circle((0, 0), rho, fill=False, ec='C2', ls='--',
                        label=fr'$\rho={rho:.3f}$'))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel('Re')
    ax.set_ylabel('Im')
    ax.set_title('(6) Closed-loop spectrum (discrete)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='lower left')

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        fig.savefig(save_path, dpi=120)
        print(f"[plots] saved {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return {'rho': rho, 'alpha': alpha, 'eig': eig, 'P': P, 'K': K}


def disturbance_rejection(data_dict, save_path=None, show=False,
                          title='Disturbance rejection (sinusoidal)'):
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))

    ax = axes[0, 0]
    for label, d in data_dict.items():
        ax.plot(d['t'], d['states'][:, 0], label=label, lw=1.3)
    ax.set_ylabel('x1 [m]')
    ax.set_xlabel('t [s]')
    ax.set_title('Main mass position under sinusoidal disturbance')
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    for label, d in data_dict.items():
        ax.plot(d['t'], d['states'][:, 1], label=label, lw=1.3)
    ax.set_ylabel('x2 [m]')
    ax.set_xlabel('t [s]')
    ax.set_title('Absorber mass position')
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    for label, d in data_dict.items():
        ax.plot(d['t'], d['controls'], label=label, lw=1.0)
    ax.set_ylabel('u(t) [N]')
    ax.set_xlabel('t [s]')
    ax.set_title('Control signal')
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=9)

    # Установившаяся RMS-амплитуда по последним 30% времени
    ax = axes[1, 1]
    labels, rms_vals, peak_vals = [], [], []
    for label, d in data_dict.items():
        t = d['t']
        x1 = d['states'][:, 0]
        mask = t > 0.5 * t[-1]
        rms = float(np.sqrt(np.mean(x1[mask] ** 2)))
        peak = float(np.max(np.abs(x1[mask])))
        labels.append(label)
        rms_vals.append(rms)
        peak_vals.append(peak)
    xpos = np.arange(len(labels))
    w = 0.35
    ax.bar(xpos - w / 2, rms_vals, w, label='steady-state RMS |x1|')
    ax.bar(xpos + w / 2, peak_vals, w, label='steady-state peak |x1|')
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_ylabel('m')
    ax.set_title('Steady-state amplitude of main mass')
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.4)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        fig.savefig(save_path, dpi=120)
        print(f"[plots] saved {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def metrics_table(data_dict, save_path=None, show=False, title='Performance metrics'):
    rows = []
    for label, d in data_dict.items():
        m = all_metrics(d, label=label)
        rows.append(m)

    headers = ['IAE_x1', 'ISE_x1', 'ITAE_x1', 'peak_x1', 'settling_x1',
               'RMS_u', 'energy_u', 'peak_u']
    cell_text = []
    row_labels = []
    for m in rows:
        row_labels.append(m['label'])
        cell_text.append([
            f"{m['IAE_x1']:.3f}",
            f"{m['ISE_x1']:.3f}",
            f"{m['ITAE_x1']:.3f}",
            f"{m['peak_x1']:.3f}",
            f"{m['settling_time_x1']:.2f}" if np.isfinite(m['settling_time_x1']) else "—",
            f"{m['RMS_u']:.3f}",
            f"{m['energy_u']:.3f}",
            f"{m['peak_u']:.3f}",
        ])

    fig, ax = plt.subplots(figsize=(12, 0.6 + 0.45 * len(rows)))
    ax.axis('off')
    tbl = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=headers,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.4)
    ax.set_title(title, pad=12)
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"[plots] saved {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return rows


# Совместимость со старым main.py — простые сводные графики
def plot_results(data, save_path=None, show=False):
    t = data['t']
    states = data['states']
    controls = data['controls']
    dist = data['disturbances']
    ref = data['ref']

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes[0, 0].plot(t, states[:, 0], label='x1')
    axes[0, 0].plot(t, states[:, 1], label='x2')
    axes[0, 0].axhline(ref[0], ls='--', color='r', label='x1_ref')
    axes[0, 0].set_ylabel('Position [m]')
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.4)

    axes[0, 1].plot(t, states[:, 2], label='v1')
    axes[0, 1].plot(t, states[:, 3], label='v2')
    axes[0, 1].set_ylabel('Velocity [m/s]')
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.4)

    axes[1, 0].plot(t, controls)
    axes[1, 0].set_ylabel('u [N]')
    axes[1, 0].set_xlabel('t [s]')
    axes[1, 0].grid(True, alpha=0.4)

    axes[1, 1].plot(t, dist)
    axes[1, 1].set_ylabel('d(t) [N]')
    axes[1, 1].set_xlabel('t [s]')
    axes[1, 1].grid(True, alpha=0.4)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    else:
        plt.close(fig)
