# src/visualization.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def euler_to_rotation_matrix(euler):
    phi, theta, psi = euler
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cps, sps = np.cos(psi), np.sin(psi)
    R = np.array([
        [cps*cth, sps*cth, -sth],
        [cps*sth*sphi - sps*cphi, sps*sth*sphi + cps*cphi, cth*sphi],
        [cps*sth*cphi + sps*sphi, sps*sth*cphi - cps*sphi, cth*cphi]
    ])
    return R


def _update(frame, data, ax):
    ax.cla()
    states = data['states']
    pos_data = states[:, 0:3]

    finite_mask = np.isfinite(pos_data)
    if not np.any(finite_mask):
        ax.text(0, 0, 0, "Симуляция нестабильна", color='red', fontsize=14, ha='center')
        return ax,

    xmin = np.nanmin(pos_data[:, 0])
    xmax = np.nanmax(pos_data[:, 0])
    ymin = np.nanmin(pos_data[:, 1])
    ymax = np.nanmax(pos_data[:, 1])
    zmin = np.nanmin(pos_data[:, 2])
    zmax = np.nanmax(pos_data[:, 2])

    # Фиксируем окно, чтобы дрон всегда был виден
    ax.set_xlim(xmin-5, xmax+5)
    ax.set_ylim(ymin-5, ymax+5)
    ax.set_zlim(max(0, zmin-2), zmax+8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Квадрокоптер | t = {data["t"][frame]:.2f} с')

    # Цель и старт
    target = data['target']
    ax.scatter(target[0], target[1], target[2], color='red', s=200, marker='*',
               label=f'Цель ({target[0]:.0f},{target[1]:.0f},{target[2]:.0f})')
    init_pos = states[0, 0:3]
    ax.scatter(init_pos[0], init_pos[1], init_pos[2], color='lime', s=140, marker='o', label='Старт')

    # Путь
    ax.plot(pos_data[:frame+1, 0], pos_data[:frame+1, 1], pos_data[:frame+1, 2],
            'b-', linewidth=3, label='Пройденный путь')

    # === ДРОН ===
    state = states[frame]
    pos = state[0:3]
    euler = state[6:9]
    u = data['controls'][frame]
    R = euler_to_rotation_matrix(euler)

    visual_l = 0.35
    motor_offsets = np.array([[ visual_l, 0, 0],
                              [ 0, visual_l, 0],
                              [-visual_l, 0, 0],
                              [ 0, -visual_l, 0]])
    motor_pos = pos + (R @ motor_offsets.T).T

    # Плечи
    for m_pos in motor_pos:
        ax.plot([pos[0], m_pos[0]], [pos[1], m_pos[1]], [pos[2], m_pos[2]],
                color='black', linewidth=7)

    # === СТРЕЛКИ ПРОПЕЛЛЕРОВ ===
    for i in range(4):
        m_pos = motor_pos[i]
        thrust_world = R @ np.array([0., 0., u[i] * 0.08])
        ax.quiver(m_pos[0], m_pos[1], m_pos[2],
                  thrust_world[0], thrust_world[1], thrust_world[2],
                  color='cyan', alpha=0.95, linewidth=2.5, arrow_length_ratio=0.4)

    # === ВЕТЕР — БОЛЬШАЯ ПОЛУПРОЗРАЧНАЯ СТРЕЛКА ===
    wind = data['winds'][frame]
    wind_pos = np.array([-14.0, -14.0, 14.0])   # фиксированная позиция для стрелки
    w_norm = np.linalg.norm(wind)
    if w_norm > 0.1:
        wind_vec = wind / w_norm * min(w_norm * 1.6, 22)
        ax.quiver(wind_pos[0], wind_pos[1], wind_pos[2],
                  wind_vec[0]*10, wind_vec[1]*10, wind_vec[2]*10,
                  color='blue', alpha=0.5, linewidth=10,
                  arrow_length_ratio=0.4, label='Ветер')

    ax.legend(loc='upper left', fontsize=10)
    return ax,


def visualize(data):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ani = FuncAnimation(fig, _update, frames=len(data['t']),
                        fargs=(data, ax), interval=20, blit=False, repeat=True)
    plt.show()