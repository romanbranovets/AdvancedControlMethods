# main.py
import numpy as np

from src.system import VerticalDrone
from src.controller import PIDController, BacksteppingController
from src.simulation import run_simulation
from src.plots import plot_results, plot_compare
from src.visualization import visualize_1d_compare
from src.plotly_dashboard import build_dashboard


def make_wind_func():
    """Скалярная функция ветра (сила в Н)."""
    def wind(t):
        return 0.3 * np.sin(2.0 * t) + 0.2 * np.sin(0.7 * t) + 0.1
    return wind


if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)

    # Генерация случайных высот
    start_z = rng.uniform(0.0, 5.0)
    target_z = rng.uniform(20.0, 30.0)
    print(f"Start height: {start_z:.2f} m")
    print(f"Target height: {target_z:.2f} m")

    # Общие параметры системы
    m = 0.5
    g = 9.81
    tau_m = 0.1
    system = VerticalDrone(m=m, g=g, tau_m=tau_m)

    wind_func = make_wind_func()

    # Начальное состояние: z0, v0=0, T0≈mg (чтобы не падал в начале)
    initial_state = np.array([start_z, 0.0, m * g])

    sim_kwargs = dict(t_max=20.0, dt=0.005, stop_tolerance=0.05)

    # ---- PID (без учёта динамики тяги) ----
    print("\n[1/2] PID simulation ...")
    ctrl_pid = PIDController(kp=5.0, ki=0.5, kd=3.0, u_max=15.0)
    data_pid = run_simulation(system, ctrl_pid, wind_func,
                              initial_state.copy(), target_z, **sim_kwargs)
    err_pid = abs(data_pid['states'][-1, 0] - target_z)
    print(f"      final error = {err_pid:.4f} m, t_end = {data_pid['t'][-1]:.2f} s")

    # ---- Backstepping ----
    print("\n[2/2] Backstepping simulation ...")
    ctrl_bs = BacksteppingController(m=m, g=g, tau_m=tau_m,
                                     kp=5.0, kd=3.0, k_T=10.0,
                                     thrust_max=15.0, u_max=15.0)
    data_bs = run_simulation(system, ctrl_bs, wind_func,
                             initial_state.copy(), target_z, **sim_kwargs)
    err_bs = abs(data_bs['states'][-1, 0] - target_z)
    print(f"      final error = {err_bs:.4f} m, t_end = {data_bs['t'][-1]:.2f} s")

    print(f"\nImprovement: {err_pid:.4f} m -> {err_bs:.4f} m  "
          f"({100*(err_pid - err_bs)/max(err_pid, 1e-9):+.1f}%)")

    # ---- Статические графики ----
    plot_results(data_pid,  save_path='results_pid.png',  show=False)
    plot_results(data_bs,   save_path='results_bs.png',   show=False)
    plot_compare(data_pid, data_bs, save_path='compare.png', show=False)
    print("Saved: results_pid.png, results_bs.png, compare.png")

    # ---- Интерактивный Plotly-дашборд ----
    print("Building interactive Plotly dashboard...")
    build_dashboard(data_pid, data_bs, target=target_z, save_path='dashboard.html')
    print("  -> open dashboard.html in a browser")

    # ---- Простая анимация (GIF) ----
    print("Creating 1D animation...")
    visualize_1d_compare(data_bs, data_pid,
                         label_main='Backstepping', label_baseline='PID',
                         target_fps=20, save_path='compare_1d.gif', show=True)