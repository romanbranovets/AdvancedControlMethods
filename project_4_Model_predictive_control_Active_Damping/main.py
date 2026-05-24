# main.py
import numpy as np
from src.system import ActiveVibrationAbsorber2D
from src.controller import MPCController, PIDController
from src.simulation import run_simulation
from src.plots import plot_results, plot_comparison, plot_trajectory_2d
from src.visualization import animate_2d, animate_mechanical_system
from src.plotly_dashboard import build_dashboard

if __name__ == "__main__":
    # Параметры двухмассовой системы
    m1 = 1.0          # основная масса
    m2 = 0.2          # масса гасителя
    k1 = 10.0         # жесткость крепления основной массы к основанию
    k2 = 5.0          # жесткость связи между массами
    c1 = 0.5          # демпфирование основной массы
    c2 = 0.1          # демпфирование связи

    # Параметры возмущения
    dist_amplitude = 2.0
    dist_freq = 3.0

    # Ограничения на управление
    u_min, u_max = -8.0, 8.0

    system = ActiveVibrationAbsorber2D(
        m1=m1, m2=m2, k1=k1, k2=k2,
        c1=c1, c2=c2,
        dist_amplitude=dist_amplitude,
        dist_freq=dist_freq,
        u_min=u_min, u_max=u_max
    )

    # Параметры MPC
    horizon = 20
    dt_control = 0.05
    # Штрафы: сильно штрафуем положение основной массы, меньше – скорость
    Q = np.diag([200.0, 10.0, 20.0, 1.0])   # x1, x2, v1, v2
    R = np.array([[1.0]])                      # скалярное управление
    mpc = MPCController(horizon=horizon, dt=dt_control, Q=Q, R=R,
                        u_min=u_min, u_max=u_max)

    # Линеаризация и дискретизация
    A, B = system.get_linear_matrices()
    Ad, Bd = system.linear_discrete(A, B, dt_control)
    mpc.set_model(Ad, Bd)

    # ПИД‑регулятор (управление по положению основной массы)
    pid = PIDController(kp=30.0, ki=8.0, kd=3.0, dt=dt_control,
                        u_min=u_min, u_max=u_max, integral_limit=3.0)

    # Начальное состояние: основная масса отклонена
    x0 = np.array([0.5, 0.0, 0.0, 0.0])
    ref = np.zeros(4)   # цель – нулевые колебания

    print("Running MPC simulation...")
    data_mpc = run_simulation(system, mpc, x0, t_max=8.0, dt=0.01, ref=ref, verbose=False)
    print("Running PID simulation...")
    data_pid = run_simulation(system, pid, x0, t_max=8.0, dt=0.01, ref=ref, verbose=False)

    # Графики
    plot_results(data_mpc, save_path='mpc_results.png', show=False)
    plot_comparison(data_mpc, data_pid, save_path='mpc_vs_pid.png', show=False)
    plot_trajectory_2d(data_mpc, data_pid, save_path='trajectory.png', show=False)

    # Анимации
    animate_2d(data_mpc, save_path='mpc_animation.gif', target_fps=25, show=False)
    # Визуализация механической системы (пружины, демпферы, две массы)
    animate_mechanical_system(data_mpc, save_path='system_animation.gif',
                              target_fps=25, show=False,
                              L0=1.5, mass_radius=0.15)

    # Дашборд
    build_dashboard(data_mpc, data_pid, save_path='dashboard.html')

    print("Done. Files saved: mpc_results.png, mpc_vs_pid.png, trajectory.png, mpc_animation.gif, system_animation.gif, dashboard.html")