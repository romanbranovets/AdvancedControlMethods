"""
simulation.py
Модуль симуляции гусеничного робота (дифференциальный привод)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
from IPython.display import HTML


def _wrap_angle(theta: float) -> float:
    """Приводит угол к диапазону [-π, π]"""
    return (theta + np.pi) % (2 * np.pi) - np.pi


def constant_command(v_l: float, v_r: float) -> Callable[[float, np.ndarray], np.ndarray]:
    u = np.array([v_l, v_r], dtype=float)

    def f(_: float, __: np.ndarray) -> np.ndarray:
        return u

    return f


def rotate_in_place(speed: float) -> Callable[[float, np.ndarray], np.ndarray]:
    u = np.array([-speed, speed], dtype=float)

    def f(_: float, __: np.ndarray) -> np.ndarray:
        return u

    return f


def go_to_pose_controller(
    target_xy: np.ndarray | Iterable[float],
    target_theta: float | None = None,
    k_rho: float = 1.5,
    k_alpha: float = 4.0,
    k_beta: float = -1.0,
    v_max: float = 1.0,
    w_max: float = 2.5,
) -> Callable[[float, np.ndarray], np.ndarray]:
    target_xy = np.asarray(target_xy, dtype=float).reshape(-1)
    if target_xy.shape != (2,):
        raise ValueError(f"expected shape (2,), got {target_xy.shape}")

    def f(_: float, state: np.ndarray) -> np.ndarray:
        x, y, th = state
        dx = target_xy[0] - x
        dy = target_xy[1] - y

        rho = np.hypot(dx, dy)
        goal_heading = np.arctan2(dy, dx)
        alpha = _wrap_angle(goal_heading - th)

        if target_theta is None:
            beta = 0.0
        else:
            beta = _wrap_angle(target_theta - goal_heading)

        v = np.clip(k_rho * rho, -v_max, v_max)
        w = np.clip(k_alpha * alpha + k_beta * beta, -w_max, w_max)

        return np.array([v, w], dtype=float)

    return f


@dataclass
class TrackedRobotSim:
    dt: float = 0.05

    body_length: float = 1.0
    body_width: float = 0.7

    track_length: float = 1.2
    track_width: float = 0.18
    track_gap: float = 0.34

    max_track_speed: float | None = None

    xlim: tuple[float, float] = (-10.0, 10.0)
    ylim: tuple[float, float] = (-10.0, 10.0)

    state: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    t: float = 0.0

    _history: list[np.ndarray] = field(default_factory=list)
    _controls: list[np.ndarray] = field(default_factory=list)
    _times: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.state = np.asarray(self.state, dtype=float).reshape(-1)
        if self.state.shape != (3,):
            raise ValueError(f"expected state shape (3,), got {self.state.shape}")
        if self.track_width <= 0 or self.track_length <= 0 or self.track_gap < 0:
            raise ValueError("track dimensions must be positive, gap must be non-negative")
        if self.body_length <= 0 or self.body_width <= 0:
            raise ValueError("body dimensions must be positive")
        self.reset(self.state)

    @property
    def track_center_distance(self) -> float:
        return self.track_gap + self.track_width

    @property
    def history(self) -> np.ndarray:
        return np.vstack(self._history)

    @property
    def controls(self) -> np.ndarray:
        return np.vstack(self._controls) if self._controls else np.empty((0, 2))

    @property
    def times(self) -> np.ndarray:
        return np.asarray(self._times, dtype=float)

    @property
    def pose(self) -> np.ndarray:
        return self.state.copy()

    def reset(self, state: np.ndarray | Iterable[float] = (0.0, 0.0, 0.0)) -> np.ndarray:
        self.state = np.asarray(state, dtype=float).reshape(-1)
        if self.state.shape != (3,):
            raise ValueError(f"expected state shape (3,), got {self.state.shape}")
        self.state[2] = _wrap_angle(self.state[2])
        self.t = 0.0
        self._history = [self.state.copy()]
        self._controls = []
        self._times = [0.0]
        return self.state.copy()

    def step_tracks(self, v_l: float, v_r: float) -> np.ndarray:
        u = np.array([v_l, v_r], dtype=float)

        if self.max_track_speed is not None:
            u = np.clip(u, -self.max_track_speed, self.max_track_speed)

        x, y, th = self.state
        L = self.track_center_distance

        v = 0.5 * (u[0] + u[1])
        w = (u[1] - u[0]) / L

        x += self.dt * v * np.cos(th)
        y += self.dt * v * np.sin(th)
        th = _wrap_angle(th + self.dt * w)

        self.state = np.array([x, y, th], dtype=float)
        self.t += self.dt

        self._history.append(self.state.copy())
        self._controls.append(u.copy())
        self._times.append(self.t)
        return self.state.copy()

    def step(self, command: np.ndarray | Iterable[float], command_mode: str = "tracks") -> np.ndarray:
        u = np.asarray(command, dtype=float).reshape(-1)
        if u.shape != (2,):
            raise ValueError(f"expected shape (2,), got {u.shape}")

        if command_mode == "tracks":
            return self.step_tracks(u[0], u[1])

        if command_mode == "vw":
            v, w = u
            L = self.track_center_distance
            v_l = v - 0.5 * L * w
            v_r = v + 0.5 * L * w
            return self.step_tracks(v_l, v_r)

        raise ValueError("command_mode must be 'tracks' or 'vw'")

    def run(
        self,
        controller: Callable[[float, np.ndarray], np.ndarray | Iterable[float]],
        steps: int,
        command_mode: str = "tracks",
    ) -> np.ndarray:
        for _ in range(steps):
            self.step(controller(self.t, self.state.copy()), command_mode=command_mode)
        return self.history

    def _body_polygons(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y, th = state
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s], [s, c]])

        def tf(poly: np.ndarray) -> np.ndarray:
            return poly @ R.T + np.array([x, y])

        bl, bw = self.body_length, self.body_width
        tl, tw, gap = self.track_length, self.track_width, self.track_gap

        body = np.array([
            [-bl / 2, -bw / 2],
            [ bl / 2, -bw / 2],
            [ bl / 2,  bw / 2],
            [-bl / 2,  bw / 2],
        ])

        yc = gap / 2 + tw / 2

        left_track = np.array([
            [-tl / 2,  yc - tw / 2],
            [ tl / 2,  yc - tw / 2],
            [ tl / 2,  yc + tw / 2],
            [-tl / 2,  yc + tw / 2],
        ])

        right_track = np.array([
            [-tl / 2, -yc - tw / 2],
            [ tl / 2, -yc - tw / 2],
            [ tl / 2, -yc + tw / 2],
            [-tl / 2, -yc + tw / 2],
        ])

        return tf(body), tf(left_track), tf(right_track)

    def plot(self, ax=None, tail: bool = True, heading: bool = True):
        hist = self.history

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 7))

        ax.set_xlim(*self.xlim)
        ax.set_ylim(*self.ylim)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        if tail and len(hist) > 1:
            ax.plot(hist[:, 0], hist[:, 1], lw=2)

        body, left_track, right_track = self._body_polygons(hist[-1])

        ax.add_patch(Polygon(left_track, closed=True, fill=False, linewidth=2))
        ax.add_patch(Polygon(right_track, closed=True, fill=False, linewidth=2))
        ax.add_patch(Polygon(body, closed=True, fill=False, linewidth=2))

        x, y, th = hist[-1]
        if heading:
            ax.arrow(
                x, y,
                0.5 * self.body_length * np.cos(th),
                0.5 * self.body_length * np.sin(th),
                head_width=0.12,
                head_length=0.18,
                length_includes_head=True,
            )

        ax.set_title(f"t = {self.t:.2f}s")
        return ax

    def animate(
        self,
        interval: int | None = None,
        tail: int | None = None,
        realtime: bool = False,
        speed: float = 1.0,
    ) -> HTML:
        hist = self.history
        if len(hist) < 2:
            raise ValueError("nothing to animate; run the simulator first")
        if speed <= 0:
            raise ValueError("speed must be > 0")

        if interval is None:
            interval = max(1, int(round(1000 * self.dt / speed))) if realtime else 40

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_xlim(*self.xlim)
        ax.set_ylim(*self.ylim)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        line, = ax.plot([], [], lw=2)
        body_patch = Polygon(np.zeros((4, 2)), closed=True, fill=False, linewidth=2)
        left_patch = Polygon(np.zeros((4, 2)), closed=True, fill=False, linewidth=2)
        right_patch = Polygon(np.zeros((4, 2)), closed=True, fill=False, linewidth=2)
        ax.add_patch(left_patch)
        ax.add_patch(right_patch)
        ax.add_patch(body_patch)

        heading_line, = ax.plot([], [], lw=2)
        title = ax.set_title("")

        def init():
            line.set_data([], [])
            heading_line.set_data([], [])
            title.set_text("")
            return line, body_patch, left_patch, right_patch, heading_line, title

        def update(i: int):
            j0 = 0 if tail is None else max(0, i - tail)
            xy = hist[j0:i + 1, :2]
            line.set_data(xy[:, 0], xy[:, 1])

            body, left_track, right_track = self._body_polygons(hist[i])
            body_patch.set_xy(body)
            left_patch.set_xy(left_track)
            right_patch.set_xy(right_track)

            x, y, th = hist[i]
            hx = x + 0.5 * self.body_length * np.cos(th)
            hy = y + 0.5 * self.body_length * np.sin(th)
            heading_line.set_data([x, hx], [y, hy])

            title.set_text(f"t = {self._times[i]:.2f}s")
            return line, body_patch, left_patch, right_patch, heading_line, title

        ani = FuncAnimation(
            fig,
            update,
            frames=len(hist),
            init_func=init,
            interval=interval,
            blit=False,
            repeat=False,
        )
        plt.close(fig)
        return HTML(ani.to_jshtml())