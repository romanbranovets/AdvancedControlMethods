# src/visualization.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle

def animate_mechanical_system(data, save_path=None, target_fps=25, show=True,
                              L0=1.5, mass_radius=0.12):
    """
    Анимация двухмассовой системы активного гасителя.
    data – словарь с ключами 't', 'states', 'ref'.
    Состояние: [x1, x2, v1, v2]
    """
    t = data['t']
    states = data['states']
    ref = data['ref']          # [x1_ref, x2_ref, ...]

    dt_sim = np.median(np.diff(t)) if len(t) > 1 else 0.02
    stride = max(1, int(round(1.0 / (target_fps * dt_sim))))
    idx = np.arange(0, len(t), stride)
    if idx[-1] != len(t) - 1:
        idx = np.append(idx, len(t) - 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(-L0 - 0.5, L0 + 0.5)
    ax.set_ylim(-0.8, 0.8)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title('Two‑mass active vibration absorber')
    ax.set_xlabel('x [m]')

    # Неподвижная левая стенка (основание)
    ax.plot([-L0, -L0], [-0.3, 0.3], 'k-', lw=4)

    # Целевые положения (звёздочки)
    ax.scatter(ref[0], 0, marker='*', c='gold', s=150, label='target x1')
    ax.scatter(ref[1], 0, marker='*', c='cyan', s=150, label='target x2')

    # Массы
    main_mass = Circle((0, 0), mass_radius, fc='red', ec='black', label='main mass')
    absorber_mass = Circle((0.5, 0), mass_radius * 0.7, fc='blue', ec='black', label='absorber')
    ax.add_patch(main_mass)
    ax.add_patch(absorber_mass)

    # Пружины (линии)
    spring_main, = ax.plot([], [], 'b-', lw=2)          # между стенкой и основной массой
    spring_coupling, = ax.plot([], [], 'g-', lw=2)      # между массами

    # Демпферы (прямоугольники)
    damper_main = Rectangle((0, 0), 0.15, 0.08, fc='gray', ec='black')
    damper_coupling = Rectangle((0, 0), 0.15, 0.08, fc='gray', ec='black')
    ax.add_patch(damper_main)
    ax.add_patch(damper_coupling)

    # Актуатор (прямоугольник между массами, оранжевый)
    actuator = Rectangle((0, 0), 0.1, 0.12, fc='orange', ec='black', alpha=0.8)
    ax.add_patch(actuator)

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10)

    ax.legend(loc='upper right')

    def draw_spring(x1, y1, x2, y2, num_coils=8, amplitude=0.05):
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        if length < 1e-6:
            return [x1, x2], [y1, y2]
        ux, uy = dx / length, dy / length
        nx, ny = -uy, ux
        tt = np.linspace(0, 1, num_coils * 4 + 2)
        xs = x1 + dx * tt + amplitude * np.sin(tt * num_coils * 2 * np.pi) * nx
        ys = y1 + dy * tt + amplitude * np.sin(tt * num_coils * 2 * np.pi) * ny
        return xs, ys

    def update(frame):
        i = idx[frame]
        x1, x2 = states[i, 0], states[i, 1]
        # Положения масс по вертикали одинаковы (y=0)
        y_mass = 0.0
        main_mass.center = (x1, y_mass)
        absorber_mass.center = (x2, y_mass)

        # Пружина от стенки до основной массы
        xs, ys = draw_spring(-L0, y_mass, x1, y_mass)
        spring_main.set_data(xs, ys)

        # Пружина между массами
        xs_c, ys_c = draw_spring(x1, y_mass, x2, y_mass, amplitude=0.04)
        spring_coupling.set_data(xs_c, ys_c)

        # Демпферы: размещаем посередине соответствующих участков
        # между стенкой и x1
        mid1 = (-L0 + x1) / 2
        damper_main.set_xy((mid1 - 0.075, y_mass - 0.04))
        # между x1 и x2
        mid2 = (x1 + x2) / 2
        damper_coupling.set_xy((mid2 - 0.075, y_mass - 0.04))

        # Актуатор: размещаем чуть выше линии соединения, между массами
        act_x = (x1 + x2) / 2 - 0.05
        act_y = y_mass + 0.2
        actuator.set_xy((act_x, act_y))

        time_text.set_text(f't = {t[i]:.2f} s')
        return (main_mass, absorber_mass, spring_main, spring_coupling,
                damper_main, damper_coupling, actuator, time_text)

    ani = FuncAnimation(fig, update, frames=len(idx), interval=1000/target_fps,
                        blit=True, repeat=True)
    if save_path:
        ani.save(save_path, writer='pillow', fps=target_fps, dpi=100)
        print(f"[viz] saved mechanical system animation to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return ani


def animate_2d(data, save_path=None, target_fps=25, show=True):
    """
    Для двухмассовой системы анимация точки в 2D не применима.
    Вызываем заглушку или перенаправляем на animate_mechanical_system.
    Чтобы main.py не падал, оставлена пустая реализация.
    """
    print("[viz] animate_2d is not applicable for 1D two‑mass system. Use animate_mechanical_system instead.")
    pass