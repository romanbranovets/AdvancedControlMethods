"""
visualization.py
Класс-обёртка для красивой визуализации гусеничного робота.
Использует готовые данные из TrackedRobotSim (история позиций + скорости гусениц).
Поддерживает анимацию пушки и снарядов, а также живой график функции Ляпунова.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Polygon
from matplotlib.lines import Line2D

# Импортируем симулятор и контроллеры
try:
    from .simulation import TrackedRobotSim, go_to_pose_controller
except ImportError:  # pragma: no cover
    from simulation import TrackedRobotSim, go_to_pose_controller


def _compute_lyapunov(
    hist: np.ndarray,
    goal: np.ndarray,
    lyapunov_c: float,
) -> np.ndarray:
    """V(ρ, α) = 0.5·ρ² + c·(1 − cos α)  for every row in hist."""
    dx = goal[0] - hist[:, 0]
    dy = goal[1] - hist[:, 1]
    rho = np.hypot(dx, dy)
    goal_heading = np.arctan2(dy, dx)
    # wrap α to (−π, π]
    alpha = np.angle(np.exp(1j * (goal_heading - hist[:, 2])))
    return 0.5 * rho ** 2 + lyapunov_c * (1.0 - np.cos(alpha))


class Visualizer:
    """Класс для визуализации движения гусеничного робота."""

    def __init__(
        self,
        sim: TrackedRobotSim,
        goal: np.ndarray | list[float] | None = None,
        obstacles: list[tuple[float, float, float]] | tuple[tuple[float, float, float], ...] | None = None,
        cannon_pos: tuple[float, float] | None = None,
        robot_radius: float = 0.35,
        goal_radius: float = 0.35,
        lyapunov_c: float = 1.0,
        title: str = "Движение гусеничного робота к цели (две гусеницы)",
        figsize: tuple[float, float] = (16, 7),
        show_controls: bool = True,
    ) -> None:
        """
        Инициализация визуализатора.

        Параметры:
            sim            — экземпляр TrackedRobotSim после sim.run(...)
            goal           — координаты цели [x, y]
            obstacles      — статические препятствия [(x, y, r), ...]
            cannon_pos     — позиция пушки (x, y), или None если пушки нет
            robot_radius   — радиус хитбокса робота, м
            lyapunov_c     — коэффициент c в V = 0.5ρ² + c(1−cosα)
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
        self.projectile_snapshots = sim.projectile_snapshots  # per-step list of (x,y,r)

        self.goal = (
            np.asarray(goal, dtype=float).reshape(2)
            if goal is not None
            else np.array([10.0, 5.0])
        )
        self.obstacles = tuple(obstacles or ())
        self.cannon_pos = cannon_pos
        self.robot_radius = robot_radius
        self.goal_radius = goal_radius
        self.lyapunov_c = lyapunov_c
        self.collision_steps = sim.collision_steps
        self.title = title
        self.figsize = figsize
        self.show_controls = show_controls

        # Предвычисляем значения функции Ляпунова для всей траектории
        self.lyapunov_values = _compute_lyapunov(self.hist, self.goal, self.lyapunov_c)

        # Для анимации (будут созданы в render)
        self.fig = None
        self.ax = None
        self.ax_lyap = None
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

        # ── Компоновка: левая панель (анимация) + правая (функция Ляпунова) ──
        self.fig = plt.figure(figsize=self.figsize)
        gs = gridspec.GridSpec(
            1, 2,
            width_ratios=[1.55, 1.0],
            wspace=0.32,
            left=0.05, right=0.97,
            top=0.88, bottom=0.09,
        )
        self.ax = self.fig.add_subplot(gs[0])
        self.ax_lyap = self.fig.add_subplot(gs[1])

        # ── Левая панель: поле ──────────────────────────────────────────────
        self.ax.set_xlim(self.sim.xlim)
        self.ax.set_ylim(self.sim.ylim)
        self.ax.set_aspect("equal")
        self.ax.grid(True, linestyle="--", alpha=0.6)
        self.ax.set_xlabel("X, м")
        self.ax.set_ylabel("Y, м")

        # Цель — хитбокс (закрашенный круг) + звезда в центре
        goal_zone = Circle(
            (self.goal[0], self.goal[1]),
            self.goal_radius,
            facecolor="#ff000033",   # полупрозрачный красный
            edgecolor="red",
            linestyle="--",
            linewidth=2.0,
            zorder=5,
            label="Цель (хитбокс)",
        )
        self.ax.add_patch(goal_zone)
        self.ax.plot(
            self.goal[0], self.goal[1], "r*", markersize=18, zorder=6
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

        # Пушка
        if self.cannon_pos is not None:
            cx, cy = self.cannon_pos
            cannon_body = Circle(
                (cx, cy), 0.45,
                facecolor="#1a1a2e",
                edgecolor="#e94560",
                linewidth=2.5,
                zorder=8,
                label="Пушка",
            )
            self.ax.add_patch(cannon_body)
            self.ax.plot(cx, cy, "r^", markersize=14, zorder=9)

        # Снаряды — создаются/удаляются каждый кадр
        self._projectile_patches: list[Circle] = []

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

        # Хитбокс танка — пунктирный круг радиуса robot_radius
        self.hitbox_patch = Circle(
            (0, 0), self.robot_radius,
            facecolor="none",
            edgecolor="cyan",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            zorder=13,
        )
        self.ax.add_patch(self.hitbox_patch)

        # Направление (красная стрелка)
        (self.heading_line,) = self.ax.plot(
            [], [], color="red", linewidth=4, solid_capstyle="butt", zorder=11
        )

        self.ax.legend(loc="upper left", fontsize=10)

        # ── Правая панель: функция Ляпунова ────────────────────────────────
        V_all = self.lyapunov_values
        t_all = self.times

        # Фоновая тень: зоны уклонения (dodge) — оранжевый
        modes_arr = self.modes
        for i in range(len(modes_arr)):
            if modes_arr[i] == "dodge" and i < len(t_all) - 1:
                self.ax_lyap.axvspan(
                    t_all[i], t_all[i + 1],
                    alpha=0.18, color="orange", linewidth=0,
                )

        # Фоновая тень: попадания — красный
        for i, hit in enumerate(self.collision_steps):
            if hit and i < len(t_all) - 1:
                self.ax_lyap.axvspan(
                    t_all[i], t_all[i + 1],
                    alpha=0.35, color="red", linewidth=0,
                )

        # Полная траектория V(t) — светло-серая
        self.ax_lyap.plot(
            t_all, V_all, color="#cccccc", linewidth=1.2, zorder=2, label="V(t) полная"
        )

        # Живая кривая — синяя (текущий прогресс)
        (self.lyap_live_line,) = self.ax_lyap.plot(
            [], [], color="#2979ff", linewidth=2.0, zorder=3, label="V(t) — текущий кадр"
        )

        # Вертикальная линия текущего времени
        self.lyap_vline = self.ax_lyap.axvline(
            x=0, color="#ff5252", linewidth=1.5, linestyle="--", zorder=4
        )

        # Текущее значение V
        self.lyap_val_text = self.ax_lyap.text(
            0.97, 0.97, "",
            transform=self.ax_lyap.transAxes,
            ha="right", va="top", fontsize=10,
            color="#2979ff",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            zorder=5,
        )

        self.ax_lyap.set_xlim(t_all[0], t_all[-1])
        self.ax_lyap.set_ylim(0, max(V_all) * 1.08 + 0.5)
        self.ax_lyap.set_xlabel("Время, с", fontsize=11)
        self.ax_lyap.set_ylabel("V(ρ, α)", fontsize=11)
        self.ax_lyap.set_title(
            "Функция Ляпунова\n"
            r"$V = \frac{1}{2}\rho^2 + c\,(1{-}\cos\alpha)$",
            fontsize=11,
        )
        self.ax_lyap.grid(True, linestyle="--", alpha=0.5)

        # Легенда: цветные вставки для режимов
        legend_handles = [
            Line2D([0], [0], color="#2979ff", linewidth=2, label="V(t)"),
            Line2D([0], [0], color="#cccccc", linewidth=2, label="V(t) полная"),
            plt.Rectangle((0, 0), 1, 1, fc="orange", alpha=0.4, label="Уклонение"),
            plt.Rectangle((0, 0), 1, 1, fc="red", alpha=0.5, label="Попадание"),
        ]
        self.ax_lyap.legend(handles=legend_handles, loc="upper right", fontsize=9)

        # ── Функция обновления ──────────────────────────────────────────────
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
                and self.modes[target_index] in ("obstacle_avoidance", "dodge")
            ):
                target = self.targets[target_index]
                self.current_target_marker.set_data([target[0]], [target[1]])
            else:
                self.current_target_marker.set_data([], [])

            # Хитбокс танка — следует за центром
            self.hitbox_patch.center = (x, y)

            # Попадание на этом кадре
            is_hit = frame < len(self.collision_steps) and self.collision_steps[frame]
            if is_hit:
                self.hitbox_patch.set_edgecolor("red")
                self.hitbox_patch.set_linewidth(3.0)
                self.body_patch.set_facecolor("#ff4444")
            else:
                self.hitbox_patch.set_edgecolor("cyan")
                self.hitbox_patch.set_linewidth(1.5)
                self.body_patch.set_facecolor("lightblue")

            # Снаряды: убираем старые, рисуем текущие
            for patch in self._projectile_patches:
                patch.remove()
            self._projectile_patches.clear()

            if frame < len(self.projectile_snapshots):
                for (px, py, pr) in self.projectile_snapshots[frame]:
                    patch = Circle(
                        (px, py), pr,
                        facecolor="#e94560",
                        edgecolor="#ff0000",
                        alpha=0.9,
                        linewidth=1.5,
                        zorder=12,
                    )
                    self.ax.add_patch(patch)
                    self._projectile_patches.append(patch)

            # ── Режим контроллера для заголовка ────────────────────────────
            mode_str = ""
            if is_hit:
                mode_str = " | \U0001f4a5 ПОПАДАНИЕ"
            elif len(self.modes) > 0:
                cur_mode = self.modes[min(frame, len(self.modes) - 1)]
                if cur_mode == "dodge":
                    mode_str = " | \u26a1 УКЛОНЕНИЕ"
                elif cur_mode == "obstacle_avoidance":
                    mode_str = " | \u21a9 обход препятствия"

            if self.show_controls and frame < len(self.controls):
                vl, vr = self.controls[frame]
                title_str = (
                    f"{self.title}\n"
                    f"t = {self.times[frame]:.2f} с | "
                    f"vₗ = {vl:+.2f} м/с | vᵣ = {vr:+.2f} м/с"
                    f"{mode_str}"
                )
            else:
                title_str = f"{self.title}\nt = {self.times[frame]:.2f} с{mode_str}"

            self.ax.set_title(title_str, fontsize=12)

            # ── Обновляем живой график V(t) ─────────────────────────────────
            t_now = self.times[frame]
            self.lyap_live_line.set_data(
                self.times[: frame + 1], V_all[: frame + 1]
            )
            self.lyap_vline.set_xdata([t_now, t_now])

            V_now = V_all[frame]
            self.lyap_val_text.set_text(f"V = {V_now:.3f}")

            # Закрыть по окончании
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
                self.hitbox_patch,
                self.lyap_live_line,
                self.lyap_vline,
                self.lyap_val_text,
                *self._projectile_patches,
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
