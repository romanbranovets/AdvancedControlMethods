# main.py
import numpy as np

from src.system import PlanarDrone2D
from src.controller import (
    BacksteppingVelocityController,
    PVelocityMotorController,
    PDVelocityMotorController,
    PIDVelocityMotorController,
)
from src.simulation import run_simulation_2d
<<<<<<< HEAD
from src.plots import (
    plot_results_2d,
    plot_velocity_controllers_comparison,
    plot_overshoot_comparison,
    plot_lyapunov_certificate,
)
=======
from src.plots import plot_results_2d_extended, plot_compare_2d_extended
>>>>>>> 6d9e6e43feea65ef680f761bfaefad7378ce5ebb
from src.visualization import visualize_2d_drone
from src.plotly_dashboard import build_dashboard_2d


def wind_field_2d(t):
    """Wind forces [N] and small external pitch torque [N·m]."""
    wx = 0.12 * np.sin(0.85 * t) + 0.05
    wz = 0.08 * np.cos(1.1 * t)
    w_tau = 0.008 * np.sin(2.2 * t)
    return np.array([wx, wz, w_tau], dtype=float)


if __name__ == "__main__":
    rng = np.random.default_rng(seed=7)

    m = 0.5
    g = 9.81
    J = 0.014
    L_arm = 0.18
    k_F = 0.38
    tau_m = 0.07

    system = PlanarDrone2D(
        m=m, g=g, J=J, L_arm=L_arm, k_F=k_F, tau_m=tau_m,
        I_min=0.0, I_max=16.0, c_d=0.06,
    )

    I0 = PlanarDrone2D.hover_currents(m, g, k_F)
    x0 = float(rng.uniform(-0.5, 0.5))
    z0 = float(rng.uniform(2.0, 5.0))
    initial_state = np.array([x0, z0, 0.0, 0.0, 0.0, 0.0, I0[0], I0[1]], dtype=float)

    target_pos = np.array([6.0, 4.0], dtype=float)
    v_des = np.zeros(2, dtype=float)

    # Same horizon for fair comparison (no early stop).
    sim_kwargs = dict(
        t_max=18.0,
        dt=0.004,
        vel_tol=0.06,
        pos_tol=0.07,
        hold_steps=100,
        verbose=False,
        early_stop=False,
        target_pos=target_pos,
        pos_gain=0.7,
        v_max=1.35,
    )

    k_th, k_w = 9.0, 5.0
    a_lim, th_lim = 6.0, 0.45
    I_min, I_max = 0.0, 16.0

    ctrl_p = PVelocityMotorController(
        m, g, J, L_arm, k_F, kp=3.0, k_th=k_th, k_w=k_w,
        a_lim=a_lim, theta_lim=th_lim, I_min=I_min, I_max=I_max,
    )
    ctrl_pd = PDVelocityMotorController(
        m, g, J, L_arm, k_F, kp=2.85, kd=0.38, k_th=k_th, k_w=k_w,
        a_lim=a_lim, theta_lim=th_lim, I_min=I_min, I_max=I_max,
    )
    ctrl_pid = PIDVelocityMotorController(
        m, g, J, L_arm, k_F, kp=2.75, kd=0.35, ki=0.22, k_th=k_th, k_w=k_w,
        a_lim=a_lim, theta_lim=th_lim, I_min=I_min, I_max=I_max,
        integral_limit=3.0,
    )
    ctrl_bs = BacksteppingVelocityController(
        m=m, g=g, J=J, L_arm=L_arm, k_F=k_F, tau_m=tau_m,
        kv=2.4, k_theta=5.0, k_omega=4.0, k_I=14.0, k_tau_blend=0.12,
        a_lim=5.5, theta_lim=0.42, I_min=I_min, I_max=I_max,
    )

    print("2D planar drone - target-point tracking (P / PD / PID / Backstepping)")
    print(
        f"  start (x,z)=({x0:.2f}, {z0:.2f}) m, "
        f"target=({target_pos[0]:.2f}, {target_pos[1]:.2f}) m"
    )

    controllers = [
        ("P", ctrl_p),
        ("PD", ctrl_pd),
        ("PID", ctrl_pid),
        ("Backstepping", ctrl_bs),
    ]

<<<<<<< HEAD
    datasets = []
    labels = []
    for name, ctrl in controllers:
        print(f"  running {name} ...")
        d = run_simulation_2d(
            system, ctrl, wind_field_2d,
            initial_state.copy(), v_des, **sim_kwargs,
        )
        epf = np.linalg.norm(d['states'][-1, 0:2] - target_pos)
        print(f"    final ||p-p*|| = {epf:.4f} m, t_end = {d['t'][-1]:.2f} s")
        datasets.append(d)
        labels.append(name)

    plot_velocity_controllers_comparison(
        datasets, labels, save_path='controllers_comparison.png', show=False,
    )
    plot_overshoot_comparison(
        datasets, labels, save_path='overshoot_comparison.png', show=False,
    )
    plot_results_2d(datasets[-1], save_path='results_2d_backstepping.png', show=False)
    plot_lyapunov_certificate(datasets[-1], save_path='lyapunov_certificate.png', show=False)
    print(
        "Saved: controllers_comparison.png, overshoot_comparison.png, "
        "results_2d_backstepping.png, lyapunov_certificate.png"
    )

    # Plotly: compare PD vs Backstepping trajectories (representative linear controllers vs BS)
    build_dashboard_2d(datasets[1], datasets[3], label_pd='PD', label_bs='Backstepping',
                       save_path='dashboard_2d.html')
    print("  -> open dashboard_2d.html in a browser")
=======
    # Расширенные графики для одного контроллера (Backstepping)
    plot_results_2d_extended(data_bs, save_path='results_2d_bs_ext.png', show=False)
    print("Saved: results_2d_bs_ext.png")

    # Сравнение двух контроллеров по всем параметрам
    plot_compare_2d_extended(data_pd, data_bs,
                             label_pd='PD+inv', label_bs='Backstepping',
                             save_path='compare_2d_ext.png', show=False)
    print("Saved: compare_2d_ext.png")

    # Интерактивная панель и анимация (без изменений)
    build_dashboard_2d(data_pd, data_bs, save_path='dashboard_2d.html')
    print("Saved: dashboard_2d.html")
>>>>>>> 6d9e6e43feea65ef680f761bfaefad7378ce5ebb

    print("Creating 2D animation (Backstepping)...")
    visualize_2d_drone(datasets[3], L_body=0.34, target_fps=22,
                       save_path='drone_2d.gif', show=False)
    print("Saved: drone_2d.gif")