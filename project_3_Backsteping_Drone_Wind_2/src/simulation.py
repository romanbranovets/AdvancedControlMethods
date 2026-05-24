# src/simulation.py
import numpy as np
from src.system import VerticalDrone, PlanarDrone2D
from src.controller import BacksteppingVelocityController


def rk4_step(system, state, t, dt, u):
    """Один шаг RK4 для системы с методом dynamics(state, t, u)."""
    def f(s, tt):
        return system.dynamics(s, tt, u)
    k1 = f(state, t)
    k2 = f(state + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(state + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(state + dt * k3, t + dt)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def run_simulation(system: VerticalDrone, controller, initial_state, target_z,
                   t_max=15.0, dt=0.005, stop_tolerance=0.05, verbose=True):
    """1D симуляция (без ветра)."""
    t = 0.0
    state = initial_state.astype(float).copy()
    controller.reset()
    times, states, controls = [], [], []
    n_steps = int(np.ceil(t_max / dt))
    for _ in range(n_steps):
        u = controller.update(state, target_z, dt)
        state = rk4_step(system, state, t, dt, u)
        t += dt
        times.append(t)
        states.append(state.copy())
        controls.append(u)
        if not np.all(np.isfinite(state)):
            if verbose:
                print(f"[sim] non-finite state at t={t:.2f}s — aborting.")
            break
        if abs(state[0] - target_z) < stop_tolerance:
            if verbose:
                print(f"[sim] target reached at t={t:.2f}s")
            break
    return {
        't': np.array(times),
        'states': np.array(states),
        'controls': np.array(controls),
        'target': target_z,
    }


def run_simulation_2d(
    system: PlanarDrone2D,
    controller,
    initial_state,
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
    Planar drone simulation. Для BacksteppingVelocityController используется
    внутренний позиционный контур, для остальных – внешний (передаётся v_ref).
    """
    t = 0.0
    state = np.asarray(initial_state, dtype=float).copy().reshape(8)
    target_pos = None if target_pos is None else np.asarray(target_pos, dtype=float).reshape(2)
    I_hover = PlanarDrone2D.hover_currents(system.m, system.g, system.k_F)

    is_backstepping = isinstance(controller, BacksteppingVelocityController)

    if is_backstepping:
        # Передаём начальное состояние и целевую позицию, чтобы правильно
        # инициализировать предыдущие значения виртуальных управлений
        controller.reset(initial_state=state, target_pos=target_pos)
    else:
        if hasattr(controller, 'reset'):
            controller.reset(I_hover=I_hover)

    times, states, controls, taus, accels = [], [], [], [], []
    v_refs = []
    lyap_V = []
    lyap_terms_rows = []
    desired_records = []   # для желаемых величин Backstepping

    n_steps = int(np.ceil(t_max / dt))
    good = 0

    for _ in range(n_steps):
        x, z, vx, vz, theta, omega, I_L, I_R = state
        T, tau = system.thrust_torque(I_L, I_R)

        # Вычисляем ускорение из динамики (без ветра)
        st, ct = np.sin(theta), np.cos(theta)
        ax = (T * st) / system.m - (system.c_d / system.m) * vx
        az = (T * ct - system.m * system.g) / system.m - (system.c_d / system.m) * vz
        a_meas = np.array([ax, az], dtype=float)

        if is_backstepping:
            # Для Backstepping передаём целевую позицию, состояние, ускорение, момент
            u = controller.update(state, target_pos, a_meas, tau, dt)
            # Сохраняем желаемые величины
            desired = controller.get_desired()
            desired_records.append(desired)
            # Для критерия остановки используем позиционную ошибку
            if target_pos is not None:
                ep = np.linalg.norm(state[0:2] - target_pos)
                is_good = ep < pos_tol and np.linalg.norm(state[2:4]) < vel_tol
            else:
                is_good = False
        else:
            # Для остальных контроллеров – внешний позиционный контур
            if target_pos is None:
                v_ref = np.zeros(2, dtype=float)
            else:
                pos_error = target_pos - np.array([x, z])
                v_ref = pos_gain * pos_error
                v_norm = float(np.linalg.norm(v_ref))
                if v_norm > v_max and v_norm > 0.0:
                    v_ref *= v_max / v_norm
            u = controller.update(v_ref, np.array([vx, vz]), a_meas, tau, np.array([I_L, I_R]), dt, theta, omega)
            v_refs.append(v_ref.copy())
            # Критерий остановки
            ev = np.linalg.norm(state[2:4] - v_ref)
            ep = np.inf if target_pos is None else np.linalg.norm(state[0:2] - target_pos)
            is_good = ev < vel_tol if target_pos is None else (ep < pos_tol and np.linalg.norm(state[2:4]) < vel_tol)

        # Шаг интегрирования
        state = rk4_step(system, state, t, dt, u)
        t += dt

        times.append(t)
        states.append(state.copy())
        controls.append(u.copy())
        taus.append(tau)
        accels.append(a_meas.copy())

        # Lyapunov для Backstepping
        lv = getattr(controller, 'last_lyapunov', None)
        lyap_V.append(np.nan if lv is None else float(lv))
        lt = getattr(controller, 'last_lyapunov_terms', None)
        if lt is None:
            lyap_terms_rows.append({'V_vel': np.nan, 'V_pitch': np.nan, 'V_omega': np.nan, 'V_current': np.nan})
        else:
            lyap_terms_rows.append({k: float(lt[k]) for k in ('V_vel', 'V_pitch', 'V_omega', 'V_current')})

        if not np.all(np.isfinite(state)):
            if verbose:
                print(f"[sim2d] non-finite state at t={t:.2f}s — aborting.")
            break

        if early_stop and is_good:
            good += 1
        else:
            good = 0
        if good >= hold_steps:
            if verbose:
                if target_pos is not None:
                    print(f"[sim2d] target point reached at t={t:.2f}s")
                else:
                    print(f"[sim2d] velocity target held at t={t:.2f}s")
            break

    lyap_terms = {k: np.array([row[k] for row in lyap_terms_rows], dtype=float)
                  for k in ('V_vel', 'V_pitch', 'V_omega', 'V_current')}

    result = {
        't': np.array(times),
        'states': np.array(states),
        'controls': np.array(controls),
        'taus': np.array(taus),
        'accels': np.array(accels),
        'v_des': np.array(v_refs) if not is_backstepping else None,
        'target_pos': None if target_pos is None else target_pos.copy(),
        'lyapunov': np.array(lyap_V, dtype=float),
        'lyapunov_terms': lyap_terms,
    }

    if is_backstepping and desired_records:
        # Преобразуем список словарей в словарь массивов
        desired_arrays = {}
        for key in desired_records[0].keys():
            desired_arrays[key] = np.array([rec[key] for rec in desired_records])
        result['desired'] = desired_arrays

    return result