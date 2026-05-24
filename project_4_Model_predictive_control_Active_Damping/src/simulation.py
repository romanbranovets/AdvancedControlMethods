# src/simulation.py
import numpy as np
from src.system import ActiveVibrationAbsorber2D  # обновлённый импорт

def rk4_step(system, state, t, dt, u):
    """Один шаг RK4."""
    def f(s, tt):
        return system.dynamics(s, tt, u)
    k1 = f(state, t)
    k2 = f(state + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(state + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(state + dt * k3, t + dt)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def run_simulation(system, controller, initial_state, t_max=10.0, dt=0.02,
                   ref=None, verbose=True):
    if ref is None:
        ref = np.zeros(4)
    else:
        ref = np.asarray(ref, dtype=float).reshape(4)

    t = 0.0
    state = initial_state.astype(float).copy()
    controller.reset()
    times = []
    states = []
    controls = []
    disturbances = []

    n_steps = int(np.ceil(t_max / dt))
    for _ in range(n_steps):
        u = controller.update(state, ref, dt)
        # u гарантированно массив, возьмём первый элемент
        u_val = np.asarray(u).flatten()[0]

        times.append(t)
        states.append(state.copy())
        controls.append(u_val)
        disturbances.append(system.disturbance(t))

        state = rk4_step(system, state, t, dt, u_val)
        t += dt

        if not np.all(np.isfinite(state)):
            if verbose:
                print(f"[sim] non-finite state at t={t:.2f}s, abort.")
            break

    return {
        't': np.array(times),
        'states': np.array(states),        # форма (N,4): x1, x2, v1, v2
        'controls': np.array(controls),    # форма (N,)
        'disturbances': np.array(disturbances),  # форма (N,)
        'ref': ref,
    }