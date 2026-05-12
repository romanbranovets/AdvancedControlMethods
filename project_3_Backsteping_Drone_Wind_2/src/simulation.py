# src/simulation.py
import numpy as np

from src.system import VerticalDrone


def rk4_step(system, state, t, dt, u, wind_func):
    """Один шаг RK4 для 1D системы."""
    v_w1 = wind_func(t) if wind_func else None
    v_w2 = wind_func(t + 0.5 * dt) if wind_func else None
    v_w4 = wind_func(t + dt) if wind_func else None

    k1 = system.dynamics(state,         t,           u, wind_func)
    k2 = system.dynamics(state + 0.5*dt*k1, t + 0.5*dt, u, wind_func)
    k3 = system.dynamics(state + 0.5*dt*k2, t + 0.5*dt, u, wind_func)
    k4 = system.dynamics(state + dt*k3,       t + dt,      u, wind_func)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def run_simulation(system: VerticalDrone,
                   controller,
                   wind_func,
                   initial_state,
                   target_z: float,
                   t_max=15.0,
                   dt=0.005,
                   stop_tolerance=0.05,
                   verbose=True):
    """
    Запуск симуляции до достижения целевой высоты с заданной точностью.

    Возвращает словарь:
        't':        массив времени
        'states':   массив состояний (N, 3)
        'controls': массив управлений (N,)
        'target':   float z_des
        'winds':    массив сил ветра (N,)
    """
    t = 0.0
    state = initial_state.astype(float).copy()
    controller.reset()

    times, states, controls, winds = [], [], [], []

    n_steps = int(np.ceil(t_max / dt))
    u = 0.0

    for _ in range(n_steps):
        w = wind_func(t) if wind_func else 0.0
        u = controller.update(state, target_z, dt)

        state = rk4_step(system, state, t, dt, u, wind_func)
        t += dt

        times.append(t)
        states.append(state.copy())
        controls.append(u)
        winds.append(w)

        if not np.all(np.isfinite(state)):
            if verbose:
                print(f"[sim] non-finite state at t={t:.2f}s — aborting.")
            break

        error = abs(state[0] - target_z)
        if error < stop_tolerance:
            if verbose:
                print(f"[sim] target reached at t={t:.2f}s, error={error:.4f}m")
            break

    return {
        't':        np.array(times),
        'states':   np.array(states),
        'controls': np.array(controls),
        'winds':    np.array(winds),
        'target':   target_z,
    }