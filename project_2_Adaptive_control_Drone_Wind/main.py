# main.py
import numpy as np

from src.system import QuadcopterSystem
from src.controller import Controller, MRACController
from src.simulation import run_simulation
from src.visualization import visualize, visualize_compare
from src.plots import plot_results, plot_compare, plot_adaptation
from src.plotly_dashboard import build_dashboard


def generate_random_points(rng, bounds=(5.0, 15.0), min_distance=6.0):
    while True:
        start = rng.uniform(*bounds, size=3)
        target = rng.uniform(*bounds, size=3)
        if np.linalg.norm(target - start) >= min_distance:
            return start, target


def make_wind_func():
    """
    Hard wind: high-frequency gust + low-frequency drift + persistent bias.
    Peak magnitude ~5 m/s. The fast component is too quick for the PID's
    integral term to follow; MRAC adapts via Phi(v) = [1, v, |v|*v].
    """
    def wind(t):
        return np.array([
            2.5 * np.sin(2.5 * t) + 1.5 * np.sin(0.4 * t) + 1.5,
            2.0 * np.cos(2.0 * t) + 1.5 * np.cos(0.3 * t) + 1.0,
            0.9 * np.sin(3.0 * t) + 0.6 * np.sin(0.5 * t) + 0.4,
        ])
    return wind


def kw_from_system(sys_):
    return dict(m=sys_.m, g=sys_.g, l=sys_.l, d=sys_.d,
                Ixx=sys_.I[0, 0], Iyy=sys_.I[1, 1], Izz=sys_.I[2, 2])


if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)

    start_pos, target_pos = generate_random_points(rng)
    print(f"Start:  {start_pos}")
    print(f"Target: {target_pos}")
    print(f"Distance: {np.linalg.norm(target_pos - start_pos):.2f} m")

    # stronger drag couples wind into translational dynamics more strongly
    system = QuadcopterSystem(c_drag_lin=0.22, c_drag_quad=0.10)
    wind_func = make_wind_func()
    initial_state = np.zeros(12)
    initial_state[0:3] = start_pos

    # stop on entering epsilon-ball; tightened to 0.15 m so PID's residual
    # tracking error (driven by fast wind) keeps it out
    sim_kwargs = dict(t_max=25.0, dt=0.005, stop_tolerance=0.15, stop_speed=None)

    # ---- run 1: PID baseline -------------------------------------------------
    print("\n[1/2] PID baseline ...")
    ctrl_pid = Controller(**kw_from_system(system))
    data_pid = run_simulation(
        system=system, controller=ctrl_pid, wind_func=wind_func,
        initial_state=initial_state, target=target_pos, **sim_kwargs,
    )
    err_pid = np.linalg.norm(data_pid['states'][-1, 0:3] - target_pos)
    print(f"      final err = {err_pid:.3f} m, t_end = {data_pid['t'][-1]:.2f} s")

    # ---- run 2: MRAC + sigma-mod --------------------------------------------
    print("\n[2/2] MRAC + sigma-modification ...")
    ctrl_mrac = MRACController(
        **kw_from_system(system),
        a_m_xy=6.0, a_m_z=8.0,
        gamma=3.0, sigma=0.5,
        theta_max=8.0, n_basis=3,
    )
    data_mrac = run_simulation(
        system=system, controller=ctrl_mrac, wind_func=wind_func,
        initial_state=initial_state, target=target_pos, **sim_kwargs,
    )
    err_mrac = np.linalg.norm(data_mrac['states'][-1, 0:3] - target_pos)
    print(f"      final err = {err_mrac:.3f} m, t_end = {data_mrac['t'][-1]:.2f} s")

    print(f"\nImprovement: {err_pid:.3f} m -> {err_mrac:.3f} m  "
          f"({100*(err_pid - err_mrac)/max(err_pid, 1e-9):+.1f}%)")

    # ---- plots ---------------------------------------------------------------
    plot_results(data_pid,  save_path='results_pid.png',  show=False)
    plot_results(data_mrac, save_path='results_mrac.png', show=False)
    plot_compare(data_pid, data_mrac, save_path='compare.png', show=False)
    plot_adaptation(ctrl_mrac.history_arrays(),
                    save_path='adaptation.png', show=False)
    print("\nSaved: results_pid.png, results_mrac.png, compare.png, adaptation.png")

    # ---- interactive Plotly dashboard ---------------------------------------
    print("Building interactive Plotly dashboard...")
    build_dashboard(data_pid, data_mrac, target_pos,
                    mrac_history=ctrl_mrac.history_arrays(),
                    gamma=3.0,
                    save_path='dashboard.html', target_fps=10)
    print("  -> open dashboard.html in a browser; drag the 3D panel during playback")

    # ---- 3D animation: PID baseline (translucent) + MRAC (solid) -----------
    print("\nLaunching side-by-side 3D animation (PID translucent + MRAC solid)...")
    visualize_compare(data_main=data_mrac, data_baseline=data_pid,
                      label_main='MRAC', label_baseline='PID',
                      target_fps=30, save_path='compare_flight.gif', show=True)
