# src/simulation.py
import numpy as np

from src.system import QuadcopterSystem
from src.controller import Controller


def rk4_step(system, state, t, dt, u, wind_func):
    v_w1 = wind_func(t)
    v_w2 = wind_func(t + 0.5 * dt)
    v_w4 = wind_func(t + dt)

    k1 = system.dynamics(state,                 t,           u, v_w1)
    k2 = system.dynamics(state + 0.5 * dt * k1, t + 0.5*dt,  u, v_w2)
    k3 = system.dynamics(state + 0.5 * dt * k2, t + 0.5*dt,  u, v_w2)
    k4 = system.dynamics(state + dt * k3,       t + dt,      u, v_w4)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def run_simulation(system: QuadcopterSystem,
                   controller: Controller,
                   wind_func,
                   initial_state: np.ndarray,
                   target: np.ndarray,
                   t_max: float = 15.0,
                   dt: float = 0.005,
                   stop_tolerance: float = 0.3,
                   stop_speed=None,
                   yaw_d: float = 0.0,
                   verbose: bool = True):
    """
    Stop conditions:
      - dist(pos, target) < stop_tolerance     (epsilon-ball reached)
      - if `stop_speed` is given (float), additionally require speed < stop_speed
    """
    t = 0.0
    state = initial_state.copy()
    controller.reset()

    times, states, controls, winds = [], [], [], []

    n_steps = int(np.ceil(t_max / dt))
    u = np.zeros(4)
    wind = wind_func(0.0)

    for _ in range(n_steps):
        wind = wind_func(t)
        u = controller.update(state, target, dt, yaw_d=yaw_d)

        times.append(t)
        states.append(state.copy())
        controls.append(u.copy())
        winds.append(np.asarray(wind, dtype=float).copy())

        state = rk4_step(system, state, t, dt, u, wind_func)
        t += dt

        if not np.all(np.isfinite(state)):
            if verbose:
                print(f"[sim] non-finite state at t={t:.2f}s — aborting.")
            break

        dist = np.linalg.norm(state[0:3] - target)
        speed = np.linalg.norm(state[3:6])
        reached = dist < stop_tolerance
        if stop_speed is not None:
            reached = reached and (speed < stop_speed)
        if reached:
            if verbose:
                print(f"[sim] target reached at t={t:.2f}s, dist={dist:.3f}m, speed={speed:.3f}m/s")
            break

    times.append(t)
    states.append(state.copy())
    controls.append(u.copy())
    winds.append(np.asarray(wind_func(t), dtype=float).copy())

    return {
        't':        np.array(times),
        'states':   np.array(states),
        'controls': np.array(controls),
        'winds':    np.array(winds),
        'target':   np.asarray(target, dtype=float),
    }
