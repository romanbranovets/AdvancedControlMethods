"""
visualization.py
Класс-обёртка для красивой визуализации гусеничного робота.
Использует готовые данные из TrackedRobotSim (история позиций + скорости гусениц).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Polygon

# Импортируем симулятор и контроллеры
try:
    from .simulation import TrackedRobotSim, go_to_pose_controller
except ImportError:  # pragma: no cover
    from simulation import TrackedRobotSim, go_to_pose_controller


class Visualizer:
    """Класс для визуализации движения гусеничного робота."""

    def __init__(
        self,
        sim: TrackedRobotSim,
        goal: np.ndarray | list[float] | None = None,
        obstacles: list[tuple[float, float, float]] | tuple[tuple[float, float, float], ...] | None = None,
        title: str = "Движение гусеничного робота к цели (две гусеницы)",
        figsize: tuple[float, float] = (12, 8),
        show_controls: bool = True,
    ) -> None:
        """
        Инициализация визуализатора.

        Параметры:
            sim            — экземпляр TrackedRobotSim после sim.run(...)
            goal           — координаты цели [x, y]
            title          — заголовок анимации
            figsize        — размер окна
            show_controls  — показывать скорости левой/правой гусеницы в заголовке
        """
        if len(sim.history) < 2:
            raise ValueError(
                "Сначала запустите симуляцию: sim.run(controller, steps=...)"
            )

        self.sim = sim
        self.hist = sim.history          # (N, 3) → [x, y, theta]
        self.controls = sim.controls     # (N-1, 2) → [v_left, v_right]
        self.times = sim.times
        self.targets = sim.targets
        self.modes = sim.modes

        self.goal = (
            np.asarray(goal, dtype=float).reshape(2)
            if goal is not None
            else np.array([10.0, 5.0])
        )
        self.obstacles = tuple(obstacles or ())
        self.title = title
        self.figsize = figsize
        self.show_controls = show_controls

        # Для анимации (будут созданы в render)
        self.fig = None
        self.ax = None
        self.ani = None

    def render(
        self,
        interval: int | None = None,
        repeat: bool = True,
        realtime: bool = True,
        close_on_finish: bool = False,
    ) -> FuncAnimation:
        """
        Запуск анимации.

        Параметры:
            interval   — задержка между кадрами в мс (None = реальное время)
            repeat     — зацикливать анимацию
            realtime   — использовать реальное время симуляции (dt)
            close_on_finish — закрыть окно после последнего кадра

        Возвращает: FuncAnimation (можно сохранить: ani.save(...))
        """
        if interval is None:
            interval = max(1, int(round(1000 * self.sim.dt))) if realtime else 40

        # Создаём фигуру и оси
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.ax.set_xlim(self.sim.xlim)
        self.ax.set_ylim(self.sim.ylim)
        self.ax.set_aspect("equal")
        self.ax.grid(True, linestyle="--", alpha=0.6)
        self.ax.set_xlabel("X, м")
        self.ax.set_ylabel("Y, м")

        # Цель
        self.ax.plot(
            self.goal[0], self.goal[1], "r*", markersize=22, label="Цель", zorder=5
        )

        for index, (x_obs, y_obs, r_obs) in enumerate(self.obstacles):
            obstacle_patch = Circle(
                (x_obs, y_obs),
                r_obs,
                facecolor="gray",
                edgecolor="black",
                alpha=0.35,
                linewidth=1.5,
                label="Препятствие" if index == 0 else None,
                zorder=3,
            )
            self.ax.add_patch(obstacle_patch)

        # Начальная позиция
        self.ax.plot(
            self.hist[0, 0],
            self.hist[0, 1],
            "go",
            markersize=12,
            label="Начальная позиция",
            zorder=5,
        )

        # Пройденный путь
        (self.path_line,) = self.ax.plot(
            [], [], "b-", linewidth=3, label="Пройденный путь", alpha=0.85, zorder=4
        )

        if len(self.targets) > 0:
            avoidance_targets = [
                target
                for target, mode in zip(self.targets, self.modes)
                if mode == "obstacle_avoidance"
            ]
            if avoidance_targets:
                unique_targets = np.unique(np.round(np.vstack(avoidance_targets), 3), axis=0)
                self.ax.scatter(
                    unique_targets[:, 0],
                    unique_targets[:, 1],
                    marker="x",
                    s=90,
                    c="orange",
                    linewidths=2.0,
                    label="Промежуточные точки",
                    zorder=6,
                )

        (self.current_target_marker,) = self.ax.plot(
            [],
            [],
            marker="o",
            markersize=9,
            color="orange",
            markeredgecolor="black",
            linestyle="None",
            label="Текущая точка обхода",
            zorder=7,
        )

        # Геометрия робота
        self.body_patch = Polygon(
            np.zeros((4, 2)),
            closed=True,
            facecolor="lightblue",
            edgecolor="darkblue",
            linewidth=2.5,
            alpha=0.95,
            zorder=10,
        )
        self.left_track_patch = Polygon(
            np.zeros((4, 2)),
            closed=True,
            facecolor="#222222",
            edgecolor="black",
            linewidth=2,
            alpha=0.9,
            zorder=9,
        )
        self.right_track_patch = Polygon(
            np.zeros((4, 2)),
            closed=True,
            facecolor="#222222",
            edgecolor="black",
            linewidth=2,
            alpha=0.9,
            zorder=9,
        )

        self.ax.add_patch(self.left_track_patch)
        self.ax.add_patch(self.right_track_patch)
        self.ax.add_patch(self.body_patch)

        # Направление (красная стрелка)
        (self.heading_line,) = self.ax.plot(
            [], [], color="red", linewidth=4, solid_capstyle="butt", zorder=11
        )

        self.ax.legend(loc="upper left", fontsize=11)

        def update(frame: int):
            state = self.hist[frame]
            x, y, theta = state

            # Геометрия робота
            body, left_track, right_track = self.sim._body_polygons(state)

            self.body_patch.set_xy(body)
            self.left_track_patch.set_xy(left_track)
            self.right_track_patch.set_xy(right_track)

            # Направление
            arrow_len = 0.5 * self.sim.body_length
            hx = x + arrow_len * np.cos(theta)
            hy = y + arrow_len * np.sin(theta)
            self.heading_line.set_data([x, hx], [y, hy])

            # Путь
            self.path_line.set_data(
                self.hist[: frame + 1, 0], self.hist[: frame + 1, 1]
            )

            target_index = min(frame, len(self.targets) - 1)
            if (
                len(self.targets) > 0
                and target_index >= 0
                and self.modes[target_index] == "obstacle_avoidance"
            ):
                target = self.targets[target_index]
                self.current_target_marker.set_data([target[0]], [target[1]])
            else:
                self.current_target_marker.set_data([], [])

            # Заголовок
            if self.show_controls and frame < len(self.controls):
                vl, vr = self.controls[frame]
                title_str = (
                    f"{self.title}\n"
                    f"t = {self.times[frame]:.2f} с | "
                    f"vₗ = {vl:+.2f} м/с | vᵣ = {vr:+.2f} м/с"
                )
            else:
                title_str = f"{self.title}\n t = {self.times[frame]:.2f} с"

            self.ax.set_title(title_str, fontsize=14)
            if close_on_finish and not repeat and frame == len(self.hist) - 1:
                self.fig.canvas.new_timer(
                    interval=max(100, interval),
                    callbacks=[(plt.close, [self.fig], {})],
                ).start()

            return (
                self.path_line,
                self.body_patch,
                self.left_track_patch,
                self.right_track_patch,
                self.heading_line,
                self.current_target_marker,
            )

        # Создаём анимацию
        self.ani = FuncAnimation(
            self.fig,
            update,
            frames=len(self.hist),
            interval=interval,
            blit=False,
            repeat=repeat,
        )

        plt.tight_layout()
        plt.show()

        return self.ani


# ====================== ПРИМЕР ИСПОЛЬЗОВАНИЯ ======================
if __name__ == "__main__":
    # Создаём симулятор
    sim = TrackedRobotSim(
        dt=0.05,
        body_length=1.0,
        body_width=0.7,
        track_length=1.2,
        track_width=0.18,
        track_gap=0.34,
        max_track_speed=3.0,
        xlim=(-2, 13),
        ylim=(-2, 8),
        state=[0.0, 0.0, 0.0],
    )

    goal = np.array([10.0, 5.0])

    # Контроллер (из simulation.py)
    controller = go_to_pose_controller(
        target_xy=goal,
        target_theta=None,
        k_rho=1.8,
        k_alpha=4.2,
        k_beta=-1.2,
        v_max=2.5,
        w_max=3.0,
    )

    print("Запуск симуляции...")
    sim.run(controller, steps=1200, command_mode="vw")

    print(f"Симуляция завершена за {len(sim.history)} шагов.")
    print(
        f"Финальная ошибка до цели: {np.hypot(sim.pose[0] - goal[0], sim.pose[1] - goal[1]):.3f} м"
    )

    # Создаём визуализатор и запускаем анимацию
    vis = Visualizer(
        sim=sim,
        goal=goal,
        title="Гусеничный робот → цель (без препятствий)",
        show_controls=True,
    )

    ani = vis.render(realtime=True, repeat=True)
    # ani.save("robot_motion.mp4", writer="ffmpeg", fps=30)  # раскомментируй для сохранения видео
