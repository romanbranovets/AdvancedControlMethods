# src/simulation.py
import numpy as np
from src.system import QuadcopterSystem
from src.controller import Controller

def rk4_step(system, state, t, dt, u, wind):
    k1 = system.dynamics(state, t, u, wind)
    k2 = system.dynamics(state + 0.5 * dt * k1, t + 0.5 * dt, u, wind)
    k3 = system.dynamics(state + 0.5 * dt * k2, t + 0.5 * dt, u, wind)
    k4 = system.dynamics(state + dt * k3, t + dt, u, wind)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def run_simulation(system: QuadcopterSystem,
                   controller: Controller,
                   wind_func,
                   initial_state: np.ndarray,
                   target: np.ndarray,
                   t_max: float = 10.0,
                   dt: float = 0.005,
                   stop_tolerance: float = 0.5):
    t = 0.0
    state = initial_state.copy()

    times = []
    states = []
    controls = []
    winds = []

    while t < t_max:
        wind = wind_func(t)
        u = controller.update(state, target, dt)

        times.append(t)
        states.append(state.copy())
        controls.append(u.copy())
        winds.append(wind.copy())

        state = rk4_step(system, state, t, dt, u, wind)

        # Защита от "убегания"
        state[6:9] = np.clip(state[6:9], -np.pi/2, np.pi/2)
        state[9:12] = np.clip(state[9:12], -100, 100)
        state[3:6] = np.clip(state[3:6], -150, 150)

        t += dt

        # Проверка достижения цели
        dist = np.linalg.norm(state[0:3] - target)
        speed = np.linalg.norm(state[3:6])
        if dist < stop_tolerance and speed < 0.5:
            print(f"Цель достигнута на t = {t:.2f} с. Расстояние: {dist:.3f} м")
            break

    # Добавляем последнее состояние после цикла
    times.append(t)
    states.append(state.copy())
    controls.append(u.copy())
    winds.append(wind.copy())

    return {
        't': np.array(times),
        'states': np.array(states),
        'controls': np.array(controls),
        'winds': np.array(winds),
        'target': target
    }