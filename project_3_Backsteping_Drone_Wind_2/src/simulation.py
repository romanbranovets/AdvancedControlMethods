# src/simulation.py
import numpy as np

from src.system import VerticalDrone, PlanarDrone2D


def rk4_step(system, state, t, dt, u, wind_func):
    """Один шаг RK4 для системы с методом dynamics(state, t, u, wind_func)."""

    def f(s, tt):
        return system.dynamics(s, tt, u, wind_func)

    k1 = f(state, t)
    k2 = f(state + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(state + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(state + dt * k3, t + dt)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


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


def run_simulation_2d(
    system: PlanarDrone2D,
    controller,
    wind_func,
    initial_state,
    v_des,
    target_pos=None,
    t_max=20.0,
    dt=0.005,
    vel_tol=0.08,
    pos_tol=0.08,
    hold_steps=80,
    verbose=True,
    early_stop=True,
    pos_gain=0.75,
    v_max=1.4,
):
    """
    Planar drone simulation. Controller: update(v_des, v, a, tau, I, dt) -> I_cmd (2,).

    State: [x, z, vx, vz, theta, omega, I_L, I_R]

    If target_pos is set, the simulator adds an outer position loop:
        v_ref = sat(pos_gain * (target_pos - [x, z]), v_max)
    so the existing velocity controllers drive the drone to a fixed point.

    If early_stop is True, stops when the active tracking error stays below tolerance.
    Records Lyapunov certificate from controller.last_lyapunov when present.
    """
    t = 0.0
    state = np.asarray(initial_state, dtype=float).copy().reshape(8)
    v_des = np.asarray(v_des, dtype=float).reshape(2)
    target_pos = None if target_pos is None else np.asarray(target_pos, dtype=float).reshape(2)
    I_hover = PlanarDrone2D.hover_currents(system.m, system.g, system.k_F)
    if hasattr(controller, 'reset'):
        controller.reset(I_hover=I_hover)

    times, states, controls, winds, taus, accels = [], [], [], [], [], []
    v_refs = []
    lyap_V = []
    lyap_terms_rows = []

    n_steps = int(np.ceil(t_max / dt))
    u = I_hover.copy()
    good = 0

    for _ in range(n_steps):
        x, z, vx, vz, theta, omega, I_L, I_R = state
        if target_pos is None:
            v_ref = v_des.copy()
        else:
            pos_error = target_pos - np.array([x, z], dtype=float)
            v_ref = pos_gain * pos_error
            v_norm = float(np.linalg.norm(v_ref))
            if v_norm > v_max and v_norm > 0.0:
                v_ref *= v_max / v_norm

        T, tau = system.thrust_torque(I_L, I_R)
        st, ct = np.sin(theta), np.cos(theta)
        ax = (T * st) / system.m - (system.c_d / system.m) * vx
        az = (T * ct - system.m * system.g) / system.m - (system.c_d / system.m) * vz
        a_vec = np.array([ax, az], dtype=float)

        if wind_func is not None:
            wpack = np.asarray(wind_func(t), dtype=float).reshape(-1)
            wx, wz = float(wpack[0]), float(wpack[1])
            w_tau = float(wpack[2]) if wpack.size >= 3 else 0.0
        else:
            wx = wz = w_tau = 0.0
        ax += wx / system.m
        az += wz / system.m

        a_meas = np.array([ax, az], dtype=float)

        u = controller.update(v_ref, np.array([vx, vz]), a_meas, tau, np.array([I_L, I_R]), dt)

        lv = getattr(controller, 'last_lyapunov', None)
        lyap_V.append(np.nan if lv is None else float(lv))
        lt = getattr(controller, 'last_lyapunov_terms', None)
        if lt is None:
            lyap_terms_rows.append(
                {'V_vel': np.nan, 'V_pitch': np.nan, 'V_omega': np.nan, 'V_current': np.nan})
        else:
            lyap_terms_rows.append({k: float(lt[k]) for k in ('V_vel', 'V_pitch', 'V_omega', 'V_current')})

        state = rk4_step(system, state, t, dt, u, wind_func)
        t += dt

        times.append(t)
        states.append(state.copy())
        controls.append(u.copy())
        winds.append(np.array([wx, wz, w_tau], dtype=float))
        taus.append(tau)
        accels.append(a_meas.copy())
        v_refs.append(v_ref.copy())

        if not np.all(np.isfinite(state)):
            if verbose:
                print(f"[sim2d] non-finite state at t={t:.2f}s — aborting.")
            break

        if early_stop:
            ev = np.linalg.norm(state[2:4] - v_ref)
            ep = np.inf if target_pos is None else np.linalg.norm(state[0:2] - target_pos)
            is_good = ev < vel_tol if target_pos is None else (ep < pos_tol and np.linalg.norm(state[2:4]) < vel_tol)
            if is_good:
                good += 1
            else:
                good = 0
            if good >= hold_steps:
                if verbose:
                    if target_pos is None:
                        print(f"[sim2d] velocity target held at t={t:.2f}s, |e_v|={ev:.4f}")
                    else:
                        print(f"[sim2d] target point reached at t={t:.2f}s, |e_p|={ep:.4f} m")
                break

    keys = ('V_vel', 'V_pitch', 'V_omega', 'V_current')
    lyap_terms = {k: np.array([row[k] for row in lyap_terms_rows], dtype=float) for k in keys}

    return {
        't': np.array(times),
        'states': np.array(states),
        'controls': np.array(controls),
        'winds': np.array(winds),
        'taus': np.array(taus),
        'accels': np.array(accels),
        'v_des': np.array(v_refs),
        'target_pos': None if target_pos is None else target_pos.copy(),
        'lyapunov': np.array(lyap_V, dtype=float),
        'lyapunov_terms': lyap_terms,
    }
