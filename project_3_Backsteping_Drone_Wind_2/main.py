# main.py
import numpy as np

from src.system import PlanarDrone2D
from src.controller import BacksteppingVelocityController, PDVelocityMotorController
from src.simulation import run_simulation_2d
from src.plots import plot_results_2d, plot_compare_2d
from src.visualization import visualize_2d_drone
from src.plotly_dashboard import build_dashboard_2d


def wind_field_2d(t):
    """Ветер (силы в Н) и слабый внешний момент по pitch [Н·м]."""
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

    v_des = np.array([0.8, 0.0], dtype=float)

    sim_kwargs = dict(t_max=18.0, dt=0.004, vel_tol=0.06, hold_steps=100, verbose=True)

    print("2D planar drone - velocity tracking")
    print(f"  start (x,z)=({x0:.2f}, {z0:.2f}) m, v_des=({v_des[0]:.2f}, {v_des[1]:.2f}) m/s")

    ctrl_pd = PDVelocityMotorController(
        m=m, g=g, J=J, L_arm=L_arm, k_F=k_F,
        kv=2.8, k_th=9.0, k_w=5.0,
        a_lim=6.0, theta_lim=0.45, I_min=0.0, I_max=16.0,
    )
    ctrl_bs = BacksteppingVelocityController(
        m=m, g=g, J=J, L_arm=L_arm, k_F=k_F, tau_m=tau_m,
        kv=2.4, k_theta=5.0, k_omega=4.0, k_I=14.0, k_tau_blend=0.12,
        a_lim=5.5, theta_lim=0.42, I_min=0.0, I_max=16.0,
    )

    print("\n[1/2] PD + static motor inversion ...")
    data_pd = run_simulation_2d(system, ctrl_pd, wind_field_2d,
                                initial_state.copy(), v_des, **sim_kwargs)
    ev_pd = np.linalg.norm(data_pd['states'][-1, 2:4] - v_des)
    print(f"      final ||v-v*|| = {ev_pd:.4f} m/s, t_end = {data_pd['t'][-1]:.2f} s")

    print("\n[2/2] Backstepping (motor dynamics) ...")
    data_bs = run_simulation_2d(system, ctrl_bs, wind_field_2d,
                                initial_state.copy(), v_des, **sim_kwargs)
    ev_bs = np.linalg.norm(data_bs['states'][-1, 2:4] - v_des)
    print(f"      final ||v-v*|| = {ev_bs:.4f} m/s, t_end = {data_bs['t'][-1]:.2f} s")

    plot_results_2d(data_bs, save_path='results_2d_bs.png', show=False)
    plot_compare_2d(data_pd, data_bs, save_path='compare_2d.png', show=False)
    print("Saved: results_2d_bs.png, compare_2d.png")

    build_dashboard_2d(data_pd, data_bs, save_path='dashboard_2d.html')
    print("  -> open dashboard_2d.html in a browser")

    print("Creating 2D animation (Backstepping)...")
    visualize_2d_drone(data_bs, L_body=0.34, target_fps=22,
                       save_path='drone_2d.gif', show=False)
    print("Saved: drone_2d.gif")
