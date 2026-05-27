# main.py
"""
Активный гаситель вибраций: строгое исследование сходимости.

Два эксперимента:
  A) Свободный отклик без возмущения — для доказательства сходимости.
     Сравниваем MPC, LQR (теоретический оптимум) и PID при одинаковых начальных
     условиях. Выводим функцию Ляпунова, спектр замкнутой системы,
     теоретическую экспоненциальную огибающую и наблюдаемую норму состояния.

  B) Подавление гармонического возмущения — для практической оценки.
     Возмущение возбуждает основную массу на её резонансной частоте.
     Сравниваем MPC (с feedforward по известному возмущению), LQR, PID
     и open-loop (управление выключено).

Артефакты:
    convergence_proof.png      — главный лист с доказательством сходимости
    disturbance_rejection.png  — графики установившегося подавления
    metrics_table.png          — таблица всех метрик
    system_animation.gif       — анимация трёх контроллеров рядом
"""
import numpy as np

from src.system import ActiveVibrationAbsorber2D
from src.controller import MPCController, LQRController, PIDController
from src.simulation import run_simulation
from src.plots import convergence_proof, disturbance_rejection, metrics_table
from src.visualization import animate_comparison
from src import scenarios


def main():
    # ---- Параметры системы --------------------------------------------------
    m1, m2 = 1.0, 0.2
    k1, k2 = 10.0, 5.0
    c1, c2 = 0.5, 0.1
    u_min, u_max = -8.0, 8.0

    system = ActiveVibrationAbsorber2D(
        m1=m1, m2=m2, k1=k1, k2=k2, c1=c1, c2=c2,
        u_min=u_min, u_max=u_max,
    )

    # ---- Дискретизация ------------------------------------------------------
    A, B, B_d = system.get_linear_matrices()
    dt_control = 0.05    # шаг управления (контроллер обновляется с этой частотой)
    dt_sim = 0.005       # шаг RK4 (в 10× быстрее — модель почти точная)
    Ad, Bd = ActiveVibrationAbsorber2D.linear_discrete(A, B, dt_control)
    _, B_d_disc = ActiveVibrationAbsorber2D.linear_discrete(A, B_d, dt_control)

    # ---- Веса штрафа --------------------------------------------------------
    # x1 (главная масса) — главный приоритет; скорости штрафуем умеренно;
    # x2 — слабо (положение гасителя само по себе не цель, важно что оно
    # вообще движется и работает).
    Q = np.diag([300.0, 5.0, 30.0, 1.0])
    R = np.array([[0.5]])

    horizon = 25

    # ---- Сценарий A: свободный отклик ---------------------------------------
    print("\n=== A) Free response (no disturbance), x0 = [0.5, 0, 0, 0] ===")
    system.set_disturbance(scenarios.no_disturbance())
    x0 = np.array([0.5, 0.0, 0.0, 0.0])
    ref = np.zeros(4)
    t_max_A = 6.0

    mpc = MPCController(horizon=horizon, dt=dt_control, Q=Q, R=R,
                        u_min=u_min, u_max=u_max, terminal_lqr=True)
    mpc.set_model(Ad, Bd)
    lqr = LQRController(Q=Q, R=R, u_min=u_min, u_max=u_max)
    lqr.set_model(Ad, Bd)
    pid = PIDController(kp=40.0, ki=10.0, kd=4.0, dt=dt_control,
                        u_min=u_min, u_max=u_max, integral_limit=3.0)

    free_data = {}
    print("  - MPC ..."); free_data['MPC'] = run_simulation(system, mpc, x0,
        t_max=t_max_A, dt=dt_sim, dt_control=dt_control, ref=ref, verbose=False)
    print("  - LQR ..."); free_data['LQR'] = run_simulation(system, lqr, x0,
        t_max=t_max_A, dt=dt_sim, dt_control=dt_control, ref=ref, verbose=False)
    print("  - PID ..."); free_data['PID'] = run_simulation(system, pid, x0,
        t_max=t_max_A, dt=dt_sim, dt_control=dt_control, ref=ref, verbose=False)

    # Доля шагов, на которых MPC попал в безусловный оптимум
    # (полезный диагностический показатель работы решателя)
    print(f"  MPC final state norm = {np.linalg.norm(free_data['MPC']['states'][-1]):.2e}")
    print(f"  LQR final state norm = {np.linalg.norm(free_data['LQR']['states'][-1]):.2e}")
    print(f"  PID final state norm = {np.linalg.norm(free_data['PID']['states'][-1]):.2e}")

    proof_info = convergence_proof(
        free_data, Ad, Bd, Q, R,
        save_path='convergence_proof.png',
        title='Convergence proof — free response, x0 = [0.5, 0, 0, 0]'
    )
    print(f"  spectral radius rho(Ad - Bd K) = {proof_info['rho']:.4f}")
    print(f"  theoretical decay rate alpha   = {proof_info['alpha']:.3f}  1/s")

    # ---- Сценарий B: гармоническое возмущение -------------------------------
    print("\n=== B) Sinusoidal disturbance (sweep through resonance area) ===")
    # Оценим собственную частоту основной массы (порядка sqrt(k1/m1)/(2π) ≈ 0.50 Гц);
    # возьмём возмущение около резонанса связанной системы — 0.7 Гц.
    dist_amp, dist_freq = 2.0, 0.7
    system.set_disturbance(scenarios.sine(dist_amp, dist_freq))

    # MPC c feedforward — знает форму возмущения
    mpc_ff = MPCController(horizon=horizon, dt=dt_control, Q=Q, R=R,
                           u_min=u_min, u_max=u_max, terminal_lqr=True,
                           dist_feedforward={'B_d': B_d_disc,
                                             'd': scenarios.sine(dist_amp, dist_freq)})
    mpc_ff.set_model(Ad, Bd)

    # Open-loop контроллер
    class _OpenLoop:
        def update(self, *a, **k): return np.array([0.0])
        def reset(self): pass

    x0B = np.zeros(4)
    ref = np.zeros(4)
    t_max_B = 10.0

    dist_data = {}
    print("  - open-loop ..."); dist_data['open-loop'] = run_simulation(
        system, _OpenLoop(), x0B, t_max=t_max_B, dt=dt_sim,
        dt_control=dt_control, ref=ref, verbose=False)
    print("  - PID ...");        dist_data['PID'] = run_simulation(
        system, pid, x0B, t_max=t_max_B, dt=dt_sim, dt_control=dt_control,
        ref=ref, verbose=False)
    print("  - LQR ...");        dist_data['LQR'] = run_simulation(
        system, lqr, x0B, t_max=t_max_B, dt=dt_sim, dt_control=dt_control,
        ref=ref, verbose=False)
    print("  - MPC+FF ...");     dist_data['MPC'] = run_simulation(
        system, mpc_ff, x0B, t_max=t_max_B, dt=dt_sim, dt_control=dt_control,
        ref=ref, verbose=False)

    disturbance_rejection(dist_data, save_path='disturbance_rejection.png',
                          title=f'Sinusoidal disturbance: A={dist_amp}, f={dist_freq} Hz')

    # ---- Таблица метрик -----------------------------------------------------
    print("\n=== Metrics (scenario A, free response) ===")
    metrics_table(free_data, save_path='metrics_free.png',
                  title='Free response — performance metrics')
    print("=== Metrics (scenario B, disturbance) ===")
    metrics_table(dist_data, save_path='metrics_disturbed.png',
                  title='Disturbance rejection — performance metrics')

    # ---- Анимации трёх контроллеров -----------------------------------------
    print("\n=== Generating animations ===")
    # Свободный отклик: возмущение off — здание возвращается к равновесию
    anim_free = {k: free_data[k] for k in ('MPC', 'LQR', 'PID')}
    animate_comparison(
        anim_free, save_path='system_animation.gif',
        target_fps=20, show=False, dpi=80,
        title='Free response — building released from initial deflection (no wind)'
    )
    # Под возмущением: ветер раскачивает здание — TMD активно его гасит
    anim_dist = {k: dist_data[k] for k in ('MPC', 'LQR', 'PID')}
    animate_comparison(
        anim_dist, save_path='system_animation_disturbed.gif',
        target_fps=20, show=False, dpi=80,
        title='Wind disturbance — TMD actively damps building sway'
    )

    print("\nDone. Files saved:")
    for f in ('convergence_proof.png', 'disturbance_rejection.png',
              'metrics_free.png', 'metrics_disturbed.png',
              'system_animation.gif', 'system_animation_disturbed.gif'):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
