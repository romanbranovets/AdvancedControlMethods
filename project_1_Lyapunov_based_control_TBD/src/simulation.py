from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

try:
    from .utils import _wrap_angle
except ImportError:  # pragma: no cover
    from utils import _wrap_angle


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


@dataclass(slots=True)
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
    _targets: list[np.ndarray] = field(default_factory=list)
    _modes: list[str] = field(default_factory=list)
    # Each entry: list of (x, y, radius) tuples for alive projectiles at that step
    _projectile_snapshots: list[list[tuple[float, float, float]]] = field(default_factory=list)
    # True at steps where a projectile actually hit the robot
    _collision_steps: list[bool] = field(default_factory=list)

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

    @property
    def targets(self) -> np.ndarray:
        return np.vstack(self._targets) if self._targets else np.empty((0, 2))

    @property
    def modes(self) -> tuple[str, ...]:
        return tuple(self._modes)

    def record_controller_debug(self, target: Iterable[float], mode: str) -> None:
        """Record controller target data for visualization."""
        target_xy = np.asarray(target, dtype=float).reshape(2)
        self._targets.append(target_xy.copy())
        self._modes.append(str(mode))

    def record_projectile_snapshot(
        self,
        projectiles: list[tuple[float, float, float]],
        robot_radius: float = 0.35,
    ) -> None:
        """Record alive projectiles and detect real hitbox collisions.

        A collision occurs when the distance between the robot centre and a
        projectile centre is less than the sum of their radii:
            |robot_pos - proj_pos| < robot_radius + proj_radius
        """
        self._projectile_snapshots.append(list(projectiles))
        robot_xy = self.state[:2]
        hit = any(
            float(np.linalg.norm(robot_xy - np.array([px, py]))) < robot_radius + pr
            for (px, py, pr) in projectiles
        )
        self._collision_steps.append(hit)

    @property
    def projectile_snapshots(self) -> list[list[tuple[float, float, float]]]:
        """Per-step list of alive projectile positions: [(x, y, r), …]."""
        return list(self._projectile_snapshots)

    @property
    def collision_steps(self) -> list[bool]:
        """Per-step flags: True when a projectile hitbox overlaps the robot."""
        return list(self._collision_steps)

    @property
    def hit_count(self) -> int:
        """Total number of simulation steps with an active collision."""
        return sum(self._collision_steps)

    def _body_polygons(self, state: np.ndarray | Iterable[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return body, left-track, and right-track polygons for visualization."""
        x, y, th = np.asarray(state, dtype=float).reshape(3)
        c = np.cos(th)
        s = np.sin(th)
        rot = np.array([[c, -s], [s, c]], dtype=float)
        origin = np.array([x, y], dtype=float)

        def rectangle(length: float, width: float, center_y: float = 0.0) -> np.ndarray:
            local = np.array(
                [
                    [0.5 * length, 0.5 * width + center_y],
                    [0.5 * length, -0.5 * width + center_y],
                    [-0.5 * length, -0.5 * width + center_y],
                    [-0.5 * length, 0.5 * width + center_y],
                ],
                dtype=float,
            )
            return local @ rot.T + origin

        track_offset = 0.5 * (self.track_gap + self.track_width)
        body = rectangle(self.body_length, self.body_width)
        left_track = rectangle(self.track_length, self.track_width, track_offset)
        right_track = rectangle(self.track_length, self.track_width, -track_offset)
        return body, left_track, right_track

    def reset(self, state: np.ndarray | Iterable[float] = (0.0, 0.0, 0.0)) -> np.ndarray:
        self.state = np.asarray(state, dtype=float).reshape(-1)
        if self.state.shape != (3,):
            raise ValueError(f"expected state shape (3,), got {self.state.shape}")
        self.state[2] = _wrap_angle(self.state[2])
        self.t = 0.0
        self._history = [self.state.copy()]
        self._controls = []
        self._times = [0.0]
        self._targets = []
        self._modes = []
        self._projectile_snapshots = []
        self._collision_steps = []
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
