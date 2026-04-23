# main.py
import numpy as np

from src.system import QuadcopterSystem
from src.controller import Controller
from src.simulation import run_simulation
from src.visualization import visualize
from src.plots import plot_results


def generate_random_points(rng, bounds=(5.0, 15.0), min_distance=6.0):
    while True:
        start = rng.uniform(*bounds, size=3)
        target = rng.uniform(*bounds, size=3)
        if np.linalg.norm(target - start) >= min_distance:
            return start, target


def make_wind_func(amp=np.array([3.0, 2.0, 1.0]),
                   freq=np.array([1.6, 1.2, 2.4])):
    """Wind as a world-frame velocity field [m/s] (sinusoidal)."""
    def wind(t):
        return np.array([
            amp[0] * np.sin(freq[0] * t),
            amp[1] * np.cos(freq[1] * t),
            amp[2] * np.sin(freq[2] * t),
        ])
    return wind


if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)

    start_pos, target_pos = generate_random_points(rng)
    print(f"Start:  {start_pos}")
    print(f"Target: {target_pos}")
    print(f"Distance: {np.linalg.norm(target_pos - start_pos):.2f} m")

    system = QuadcopterSystem()
    controller = Controller(m=system.m, g=system.g, l=system.l, d=system.d)

    initial_state = np.zeros(12)
    initial_state[0:3] = start_pos

    wind_func = make_wind_func()

    print("Running simulation (cascaded PID, wind as velocity field, drag)...")
    sim_data = run_simulation(
        system=system,
        controller=controller,
        wind_func=wind_func,
        initial_state=initial_state,
        target=target_pos,
        t_max=20.0,
        dt=0.005,
        stop_tolerance=0.3,
        stop_speed=0.3,
    )

    final_pos = sim_data['states'][-1, 0:3]
    dist = np.linalg.norm(final_pos - target_pos)
    print(f"Final position: {final_pos}")
    print(f"Final error:    {dist:.3f} m")

    plot_results(sim_data, save_path='results.png', show=False)
    print("Saved time-series plots to results.png")

    print("Launching 3D animation...")
    visualize(sim_data)
