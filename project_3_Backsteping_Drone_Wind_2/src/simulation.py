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
    t_max=20.0,
    dt=0.005,
    vel_tol=0.08,
    hold_steps=80,
    verbose=True,
):
    """
    Симуляция планарного дрона. Регулятор: update(v_des, v, a, tau, I, dt, theta, omega) -> I_cmd (2,).

    Состояние: [x, z, vx, vz, theta, omega, I_L, I_R]
    """
    t = 0.0
    state = np.asarray(initial_state, dtype=float).copy().reshape(8)
    v_des = np.asarray(v_des, dtype=float).reshape(2)
    I_hover = PlanarDrone2D.hover_currents(system.m, system.g, system.k_F)
    if hasattr(controller, 'reset'):
        controller.reset(I_hover=I_hover)

    times, states, controls, winds, taus, accels = [], [], [], [], [], []
    desired_records = []      # theta_des, omega_des, I_L_des, I_R_des, tau_des, T_des
    pos_des_records = []      # x_des, z_des

    x0, z0 = float(initial_state[0]), float(initial_state[1])

    n_steps = int(np.ceil(t_max / dt))
    u = I_hover.copy()
    good = 0

    for _ in range(n_steps):
        x, z, vx, vz, theta, omega, I_L, I_R = state
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

        # Вызов контроллера с передачей theta и omega
        u = controller.update(v_des, np.array([vx, vz]), a_meas, tau,
                              np.array([I_L, I_R]), dt, theta, omega)

        # Сохранение желаемых величин
        desired = controller.get_desired()
        desired_records.append([desired['theta_des'], desired['omega_des'],
                                desired['I_des'][0], desired['I_des'][1],
                                desired['tau_des'], desired['T_des']])

        # Желаемая позиция (интеграл от постоянной v_des)
        pos_des_records.append([x0 + v_des[0] * t, z0 + v_des[1] * t])

        state = rk4_step(system, state, t, dt, u, wind_func)
        t += dt

        times.append(t)
        states.append(state.copy())
        controls.append(u.copy())
        winds.append(np.array([wx, wz, w_tau], dtype=float))
        taus.append(tau)
        accels.append(a_meas.copy())

        if not np.all(np.isfinite(state)):
            if verbose:
                print(f"[sim2d] non-finite state at t={t:.2f}s — aborting.")
            break

        ev = np.linalg.norm(state[2:4] - v_des)
        if ev < vel_tol:
            good += 1
        else:
            good = 0
        if good >= hold_steps:
            if verbose:
                print(f"[sim2d] velocity target held at t={t:.2f}s, |e_v|={ev:.4f}")
            break

    return {
        't': np.array(times),
        'states': np.array(states),
        'controls': np.array(controls),
        'winds': np.array(winds),
        'taus': np.array(taus),
        'accels': np.array(accels),
        'v_des': v_des.copy(),
        'desired': np.array(desired_records),   # shape (N, 6)
        'pos_des': np.array(pos_des_records),   # shape (N, 2)
    }