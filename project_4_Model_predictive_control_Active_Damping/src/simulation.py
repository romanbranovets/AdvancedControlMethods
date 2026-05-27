# src/simulation.py
"""
Симуляция с физически корректным разделением частот:
    - dt_sim    — шаг интегратора (RK4), для точности модели
    - dt_control — шаг управления, контроллер вызывается ровно с этой частотой,
                   управление удерживается ZOH между вызовами.
Если dt_control не передан, контроллер обновляется каждый шаг (старое поведение).
"""
import numpy as np


def rk4_step(system, state, t, dt, u):
    def f(s, tt):
        return system.dynamics(s, tt, u)
    k1 = f(state, t)
    k2 = f(state + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(state + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(state + dt * k3, t + dt)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def run_simulation(system, controller, initial_state, t_max=10.0,
                   dt=0.01, dt_control=None, ref=None, verbose=True):
    if ref is None:
        ref = np.zeros(4)
    ref = np.asarray(ref, dtype=float).reshape(4)

    if dt_control is None:
        dt_control = dt

    t = 0.0
    state = initial_state.astype(float).copy()
    controller.reset()
    times, states, controls, disturbances = [], [], [], []

    u = np.array([0.0])
    last_ctrl_t = -np.inf
    n_steps = int(np.ceil(t_max / dt))

    for k in range(n_steps):
        # ZOH: обновляем команду только когда подошло время следующего шага управления
        if t - last_ctrl_t >= dt_control - 1e-12:
            u = controller.update(state, ref, dt_control, t=t)
            last_ctrl_t = t
        u_val = float(np.asarray(u).flatten()[0])

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
        'states': np.array(states),
        'controls': np.array(controls),
        'disturbances': np.array(disturbances),
        'ref': ref,
        'dt': dt,
        'dt_control': dt_control,
    }
