# main.py
import numpy as np
from src.system import QuadcopterSystem
from src.controller import Controller
from src.simulation import run_simulation
from src.visualization import visualize

def generate_random_points(bounds=(10, 30), min_distance=10.0):
    """Генерирует случайные начальную позицию и цель внутри куба bounds,
       с расстоянием не менее min_distance."""
    while True:
        start = np.random.uniform(*bounds, size=3)
        target = np.random.uniform(*bounds, size=3)
        if np.linalg.norm(target - start) >= min_distance:
            return start, target

if __name__ == "__main__":
    # Генерация случайных точек
    start_pos, target_pos = generate_random_points()
    print(f"Сгенерированы точки:\n  Старт: {start_pos}\n  Цель:  {target_pos}")
    print(f"Расстояние: {np.linalg.norm(target_pos - start_pos):.2f} м")

    system = QuadcopterSystem()
    controller = Controller()

    initial_state = np.zeros(12)
    initial_state[0:3] = start_pos

    def wind_func(t):
        # Переменный ветер (как в исходном примере)
        return np.array([
            1.8 * np.sin(1.6 * t),
            1.4 * np.cos(1.2 * t),
            0.6 * np.sin(2.4 * t)
        ])

    print("Запуск симуляции (каскадный PID, переменный ветер, макс. наклон 10°)...")
    sim_data = run_simulation(
        system=system,
        controller=controller,
        wind_func=wind_func,
        initial_state=initial_state,
        target=target_pos,
        t_max=5.0,          # достаточно времени для преодоления расстояния
        dt=0.005,
        stop_tolerance=0.5
    )

    final_pos = sim_data['states'][-1, 0:3]
    dist = np.linalg.norm(final_pos - target_pos)
    print(f"Симуляция завершена. Финальная позиция: {final_pos}")
    print(f"Расстояние до цели: {dist:.3f} м")

    print("Запуск 3D-анимации...")
    visualize(sim_data)