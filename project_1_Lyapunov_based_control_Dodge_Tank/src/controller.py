from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np

try:
    from .utils import wrap_to_pi
except ImportError:  # pragma: no cover
    from utils import wrap_to_pi


def _as_pose(state: object) -> tuple[float, float, float]:
    """Extract x, y, theta from common state representations."""
    if isinstance(state, Mapping):
        return float(state["x"]), float(state["y"]), float(state["theta"])

    if all(hasattr(state, name) for name in ("x", "y", "theta")):
        return float(state.x), float(state.y), float(state.theta)

    arr = np.asarray(state, dtype=float).reshape(-1)
    if arr.shape[0] < 3:
        raise ValueError(f"state must contain at least 3 values, got shape {arr.shape}")
    return float(arr[0]), float(arr[1]), float(arr[2])


def _as_goal_xy(goal: object) -> tuple[float, float]:
    """Extract x_goal, y_goal from common goal representations."""
    if isinstance(goal, Mapping):
        x_key = "x_goal" if "x_goal" in goal else "x"
        y_key = "y_goal" if "y_goal" in goal else "y"
        return float(goal[x_key]), float(goal[y_key])

    if all(hasattr(goal, name) for name in ("x_goal", "y_goal")):
        return float(goal.x_goal), float(goal.y_goal)

    if all(hasattr(goal, name) for name in ("x", "y")):
        return float(goal.x), float(goal.y)

    arr = np.asarray(goal, dtype=float).reshape(-1)
    if arr.shape[0] < 2:
        raise ValueError(f"goal must contain at least 2 values, got shape {arr.shape}")
    return float(arr[0]), float(arr[1])


@dataclass(frozen=True, slots=True)
class CircularObstacle:
    """Circular obstacle with an inflated safety radius."""

    x: float
    y: float
    radius: float

    @property
    def center(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    def inflated_radius(self, robot_radius: float, safety_margin: float) -> float:
        return self.radius + robot_radius + safety_margin


def _as_obstacle(obstacle: object) -> CircularObstacle:
    """Extract a circular obstacle from dicts, objects, or tuples."""
    if isinstance(obstacle, CircularObstacle):
        return obstacle

    if isinstance(obstacle, Mapping):
        if "center" in obstacle:
            x, y = _as_goal_xy(obstacle["center"])
        else:
            x = obstacle["x_obs"] if "x_obs" in obstacle else obstacle["x"]
            y = obstacle["y_obs"] if "y_obs" in obstacle else obstacle["y"]
        radius = obstacle["r_obs"] if "r_obs" in obstacle else obstacle["radius"]
        return CircularObstacle(float(x), float(y), float(radius))

    if all(hasattr(obstacle, name) for name in ("x_obs", "y_obs", "r_obs")):
        return CircularObstacle(
            float(obstacle.x_obs),
            float(obstacle.y_obs),
            float(obstacle.r_obs),
        )

    if all(hasattr(obstacle, name) for name in ("x", "y", "radius")):
        return CircularObstacle(
            float(obstacle.x),
            float(obstacle.y),
            float(obstacle.radius),
        )

    arr = np.asarray(obstacle, dtype=float).reshape(-1)
    if arr.shape[0] < 3:
        raise ValueError(f"obstacle must contain x, y, radius, got shape {arr.shape}")
    return CircularObstacle(float(arr[0]), float(arr[1]), float(arr[2]))


def segment_circle_intersection(
    p0: Iterable[float],
    p1: Iterable[float],
    center: Iterable[float],
    radius: float,
) -> bool:
    """Return True when a line segment intersects a circle."""
    if radius < 0.0:
        raise ValueError("radius must be non-negative")

    start = np.asarray(p0, dtype=float).reshape(2)
    end = np.asarray(p1, dtype=float).reshape(2)
    c = np.asarray(center, dtype=float).reshape(2)
    seg = end - start
    seg_len_sq = float(seg @ seg)
    if seg_len_sq <= 1e-12:
        return float(np.linalg.norm(start - c)) <= radius

    projection = float(((c - start) @ seg) / seg_len_sq)
    projection = float(np.clip(projection, 0.0, 1.0))
    closest = start + projection * seg
    return float(np.linalg.norm(closest - c)) <= radius


def compute_tangent_waypoints(
    position: Iterable[float],
    obstacle: CircularObstacle,
    inflated_radius: float,
    waypoint_margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute clockwise and counterclockwise tangent waypoints."""
    pos = np.asarray(position, dtype=float).reshape(2)
    center = obstacle.center
    rel = pos - center
    dist = float(np.linalg.norm(rel))
    radius = inflated_radius + waypoint_margin

    if dist <= radius + 1e-9:
        # Inside or too close for a geometric tangent: step sideways from the
        # radial escape direction to produce two safe waypoint candidates.
        if dist <= 1e-9:
            radial = np.array([1.0, 0.0], dtype=float)
        else:
            radial = rel / dist
        side = np.array([-radial[1], radial[0]], dtype=float)
        return center + radius * side, center - radius * side

    base = float(np.arctan2(rel[1], rel[0]))
    offset = float(np.arccos(radius / dist))
    return (
        center + radius * np.array([np.cos(base + offset), np.sin(base + offset)]),
        center + radius * np.array([np.cos(base - offset), np.sin(base - offset)]),
    )


def choose_tangent_waypoint(
    position: Iterable[float],
    theta: float,
    goal: Iterable[float],
    candidates: tuple[np.ndarray, np.ndarray],
    previous_direction: int | None = None,
) -> tuple[np.ndarray, int]:
    """Choose the tangent waypoint with lower path and heading cost."""
    pos = np.asarray(position, dtype=float).reshape(2)
    goal_xy = np.asarray(goal, dtype=float).reshape(2)

    if previous_direction in (-1, 1):
        index = 0 if previous_direction == 1 else 1
        return candidates[index].copy(), previous_direction

    best_cost = np.inf
    best_index = 0
    for index, waypoint in enumerate(candidates):
        to_waypoint = waypoint - pos
        heading = float(np.arctan2(to_waypoint[1], to_waypoint[0]))
        heading_change = abs(float(wrap_to_pi(heading - theta)))
        path_cost = float(np.linalg.norm(to_waypoint) + np.linalg.norm(goal_xy - waypoint))
        cost = path_cost + 0.25 * heading_change
        if cost < best_cost:
            best_cost = cost
            best_index = index

    direction = 1 if best_index == 0 else -1
    return candidates[best_index].copy(), direction


def compute_side_bypass_waypoints(
    position: Iterable[float],
    goal: Iterable[float],
    obstacle: CircularObstacle,
    bypass_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute two simple side waypoints around a blocking obstacle."""
    pos = np.asarray(position, dtype=float).reshape(2)
    goal_xy = np.asarray(goal, dtype=float).reshape(2)
    path = goal_xy - pos
    path_length = float(np.linalg.norm(path))
    if path_length <= 1e-9:
        direction = np.array([1.0, 0.0], dtype=float)
    else:
        direction = path / path_length
    normal = np.array([-direction[1], direction[0]], dtype=float)

    # Put the waypoint slightly past the obstacle, not exactly beside it. This
    # makes the Lyapunov target stable and avoids orbiting around the circle.
    base = obstacle.center + 0.6 * bypass_radius * direction
    return base + bypass_radius * normal, base - bypass_radius * normal


@dataclass(frozen=True, slots=True)
class MovingObstacle:
    """A moving circular obstacle (e.g. a projectile).

    Attributes
    ----------
    x, y     : current position, m
    vx, vy   : current velocity, m/s  (assumed constant over the lookahead horizon)
    radius   : collision radius, m
    """

    x: float
    y: float
    vx: float
    vy: float
    radius: float

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy], dtype=float)


@dataclass(slots=True)
class ControlDiagnostics:
    """Diagnostic values for the Lyapunov point-stabilizing controller."""

    mode: str
    current_target: tuple[float, float]
    blocking_obstacle_index: int | None
    rho: float
    alpha: float
    v: float
    omega: float
    u_l: float
    u_r: float
    V: float
    avoidance_active: bool


@dataclass(slots=True)
class LyapunovTrackedRobotController:
    """Lyapunov controller for a tracked robot modeled as a unicycle.

    The controller stabilizes the robot position to a target point. The final
    heading is intentionally left unconstrained.
    """

    goal: Iterable[float] | Mapping[str, float]
    k_rho: float = 1.5
    k_alpha: float = 4.0
    b: float = 0.52
    u_max: float | None = 2.0
    eps_goal: float = 0.05
    lyapunov_c: float = 1.0
    output_mode: str = "tracks"
    obstacles: Iterable[object] = field(default_factory=tuple)
    robot_radius: float = 0.35
    safety_margin: float = 0.15
    obstacle_activation_distance: float = 0.5
    obstacle_clearance_hysteresis: float = 0.25
    waypoint_margin: float = 1.0
    unsafe_turn_rate: float = 1.0
    min_avoidance_speed: float = 0.25
    max_forward_heading_error: float = 1.25
    stall_position_epsilon: float = 0.02
    stall_goal_progress_epsilon: float = 0.005
    stall_steps: int = 20
    stall_waypoint_margin_boost: float = 0.6
    visibility_samples_per_obstacle: int = 12
    planner_clearance: float = 0.35
    waypoint_reached_radius: float = 0.35
    # Dodge parameters for moving obstacles (projectiles)
    dodge_lookahead: float = 1.2        # seconds to look ahead for collision
    dodge_danger_factor: float = 1.4   # react to near-misses within 40% buffer
    obstacle_avoidance_active: bool = field(init=False, default=False)
    current_waypoint: np.ndarray | None = field(init=False, default=None)
    current_obstacle_index: int | None = field(init=False, default=None)
    tangent_direction: int | None = field(init=False, default=None)
    last_position: np.ndarray | None = field(init=False, default=None)
    last_goal_distance: float | None = field(init=False, default=None)
    stall_counter: int = field(init=False, default=0)
    waypoint_margin_boost_active: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.goal = np.array(_as_goal_xy(self.goal), dtype=float)
        self.obstacles = tuple(_as_obstacle(obstacle) for obstacle in self.obstacles)
        if self.k_rho <= 0.0:
            raise ValueError("k_rho must be positive")
        if self.k_alpha <= self.k_rho:
            raise ValueError("k_alpha must be greater than k_rho")
        if self.b <= 0.0:
            raise ValueError("b must be positive")
        if self.u_max is not None and self.u_max <= 0.0:
            raise ValueError("u_max must be positive when provided")
        if self.eps_goal < 0.0:
            raise ValueError("eps_goal must be non-negative")
        if self.lyapunov_c <= 0.0:
            raise ValueError("lyapunov_c must be positive")
        if self.output_mode not in ("tracks", "vw"):
            raise ValueError("output_mode must be 'tracks' or 'vw'")
        if self.robot_radius < 0.0:
            raise ValueError("robot_radius must be non-negative")
        if self.safety_margin < 0.0:
            raise ValueError("safety_margin must be non-negative")
        if self.obstacle_activation_distance < 0.0:
            raise ValueError("obstacle_activation_distance must be non-negative")
        if self.obstacle_clearance_hysteresis < 0.0:
            raise ValueError("obstacle_clearance_hysteresis must be non-negative")
        if self.waypoint_margin < 0.0:
            raise ValueError("waypoint_margin must be non-negative")
        if self.unsafe_turn_rate <= 0.0:
            raise ValueError("unsafe_turn_rate must be positive")
        if self.min_avoidance_speed < 0.0:
            raise ValueError("min_avoidance_speed must be non-negative")
        if self.max_forward_heading_error <= 0.0:
            raise ValueError("max_forward_heading_error must be positive")
        if self.stall_position_epsilon < 0.0:
            raise ValueError("stall_position_epsilon must be non-negative")
        if self.stall_goal_progress_epsilon < 0.0:
            raise ValueError("stall_goal_progress_epsilon must be non-negative")
        if self.stall_steps < 1:
            raise ValueError("stall_steps must be at least 1")
        if self.stall_waypoint_margin_boost < 0.0:
            raise ValueError("stall_waypoint_margin_boost must be non-negative")
        if self.visibility_samples_per_obstacle < 4:
            raise ValueError("visibility_samples_per_obstacle must be at least 4")
        if self.planner_clearance < 0.0:
            raise ValueError("planner_clearance must be non-negative")
        if self.waypoint_reached_radius <= 0.0:
            raise ValueError("waypoint_reached_radius must be positive")
        if self.dodge_lookahead <= 0.0:
            raise ValueError("dodge_lookahead must be positive")
        if self.dodge_danger_factor <= 0.0:
            raise ValueError("dodge_danger_factor must be positive")

    def virtual_control(
        self,
        state: object,
        goal: object | None = None,
        obstacles: Iterable[object] | None = None,
        moving_obstacles: Iterable[MovingObstacle] | None = None,
    ) -> tuple[float, float]:
        """Return the equivalent unicycle command (v, omega)."""
        diagnostics = self.diagnostics(state, goal, obstacles, moving_obstacles)
        return diagnostics.v, diagnostics.omega

    def track_control(
        self,
        state: object,
        goal: object | None = None,
        obstacles: Iterable[object] | None = None,
        moving_obstacles: Iterable[MovingObstacle] | None = None,
    ) -> tuple[float, float]:
        """Return left and right track velocities (u_L, u_R)."""
        diagnostics = self.diagnostics(state, goal, obstacles, moving_obstacles)
        return diagnostics.u_l, diagnostics.u_r

    def get_control(
        self,
        state: object,
        goal: object | None = None,
        obstacles: Iterable[object] | None = None,
        moving_obstacles: Iterable[MovingObstacle] | None = None,
    ) -> np.ndarray:
        """Return a command compatible with the configured output mode."""
        diagnostics = self.diagnostics(state, goal, obstacles, moving_obstacles)
        if self.output_mode == "vw":
            return np.array([diagnostics.v, diagnostics.omega], dtype=float)
        return np.array([diagnostics.u_l, diagnostics.u_r], dtype=float)

    def get_control_with_debug(
        self,
        state: object,
        goal: object | None = None,
        obstacles: Iterable[object] | None = None,
        moving_obstacles: Iterable[MovingObstacle] | None = None,
    ) -> tuple[float, float, ControlDiagnostics]:
        """Return track commands and debug information."""
        diagnostics = self.diagnostics(state, goal, obstacles, moving_obstacles)
        return diagnostics.u_l, diagnostics.u_r, diagnostics

    def __call__(self, _: float, state: object) -> np.ndarray:
        """Simulation callback with signature controller(t, state)."""
        return self.get_control(state)

    def diagnostics(
        self,
        state: object,
        goal: object | None = None,
        obstacles: Iterable[object] | None = None,
        moving_obstacles: Iterable[MovingObstacle] | None = None,
    ) -> ControlDiagnostics:
        """Compute command and Lyapunov diagnostic values.

        Priority order:
          1. Unsafe recovery  (robot inside a static obstacle — numerical edge case)
          2. Dodge manoeuvre  (incoming projectile within lookahead horizon)
          3. Static obstacle avoidance / normal goal tracking
        """
        x, y, theta = _as_pose(state)
        goal_xy = np.array(_as_goal_xy(self.goal if goal is None else goal), dtype=float)
        obstacle_list = self.obstacles if obstacles is None else tuple(
            _as_obstacle(obstacle) for obstacle in obstacles
        )

        pos = np.array([x, y], dtype=float)

        # Priority 1: dodge incoming projectile — checked first so a bullet threat
        # is never ignored even when the robot is inside a static obstacle.
        if moving_obstacles is not None:
            dodge_diag = self._dodge_check(pos, theta, tuple(moving_obstacles))
            if dodge_diag is not None:
                return dodge_diag

        # Priority 2: unsafe recovery (robot clipped inside a static obstacle)
        unsafe = self._unsafe_obstacle(pos, obstacle_list)
        if unsafe is not None:
            index, obstacle = unsafe
            diagnostics = self._unsafe_recovery(pos, theta, obstacle, index)
            self.obstacle_avoidance_active = True
            self.current_obstacle_index = index
            return diagnostics

        # Priority 3: normal navigation (static obstacle avoidance or goal tracking)
        stalled = self._update_stall_state(pos, goal_xy)
        target, mode, blocking_index = self._select_current_target(pos, theta, goal_xy, obstacle_list)
        return self._lyapunov_diagnostics(
            state=(x, y, theta),
            target=target,
            mode=mode,
            blocking_obstacle_index=blocking_index,
            stalled=stalled,
        )

    def _lyapunov_diagnostics(
        self,
        state: tuple[float, float, float],
        target: np.ndarray,
        mode: str,
        blocking_obstacle_index: int | None,
        stalled: bool = False,
    ) -> ControlDiagnostics:
        """Apply the Lyapunov point controller to the selected target."""
        x, y, theta = state
        dx = target[0] - x
        dy = target[1] - y
        rho = float(np.hypot(dx, dy))
        phi = float(np.arctan2(dy, dx)) if rho > 0.0 else theta
        alpha = float(wrap_to_pi(phi - theta))

        if rho < self.eps_goal:
            v = 0.0
            omega = 0.0
        else:
            v = float(self.k_rho * rho * np.cos(alpha))
            omega = float(self.k_alpha * np.sin(alpha))
            # Near-goal damping: scale ω by ρ/(ρ+ε) so that as ρ→0, ω→0.
            # This prevents noise-induced spinning when the robot is close to
            # the target.  The standard Lyapunov analysis holds for ρ > eps_goal;
            # inside eps_goal the robot stops anyway.
            omega_scale = rho / (rho + 3.0 * self.eps_goal)
            omega *= omega_scale
            if mode == "obstacle_avoidance":
                v, omega = self._avoidance_motion_guard(v, omega, alpha, stalled)

        u_l, u_r = unicycle_to_tracks(v, omega, self.b)
        if self.u_max is not None:
            u_l = float(np.clip(u_l, -self.u_max, self.u_max))
            u_r = float(np.clip(u_r, -self.u_max, self.u_max))
            v, omega = tracks_to_unicycle(u_l, u_r, self.b)

        V = 0.5 * rho**2 + self.lyapunov_c * (1.0 - float(np.cos(alpha)))
        return ControlDiagnostics(
            mode=mode,
            current_target=(float(target[0]), float(target[1])),
            blocking_obstacle_index=blocking_obstacle_index,
            rho=rho,
            alpha=alpha,
            v=float(v),
            omega=float(omega),
            u_l=float(u_l),
            u_r=float(u_r),
            V=float(V),
            avoidance_active=self.obstacle_avoidance_active,
        )

    def _select_current_target(
        self,
        pos: np.ndarray,
        theta: float,
        goal: np.ndarray,
        obstacles: tuple[CircularObstacle, ...],
    ) -> tuple[np.ndarray, str, int | None]:
        """Select either the true goal or a temporary tangent waypoint."""
        if not obstacles:
            self._clear_avoidance()
            return goal, "goal_tracking", None

        blocking = self._nearest_blocking_obstacle(pos, goal, obstacles)
        if (
            self.obstacle_avoidance_active
            and self.current_obstacle_index is not None
            and self.current_obstacle_index < len(obstacles)
        ):
            clear_blocking = self._nearest_blocking_obstacle(
                pos,
                goal,
                obstacles,
                extra_radius=self.obstacle_clearance_hysteresis,
                include_activation_zone=False,
            )
            if clear_blocking is None:
                self._clear_avoidance()
                return goal, "goal_tracking", None
            blocking = (
                self.current_obstacle_index,
                obstacles[self.current_obstacle_index],
            )

        if blocking is None:
            self._clear_avoidance()
            return goal, "goal_tracking", None

        index, obstacle = blocking
        if self.current_waypoint is not None:
            waypoint_distance = float(np.linalg.norm(self.current_waypoint - pos))
            waypoint_is_reachable = self._segment_is_clear(pos, self.current_waypoint, obstacles)
            if waypoint_distance > self.waypoint_reached_radius and waypoint_is_reachable:
                self.obstacle_avoidance_active = True
                return self.current_waypoint.copy(), "obstacle_avoidance", index
            self.current_waypoint = None

        planned_waypoint = self._simple_bypass_waypoint(pos, theta, goal, obstacle, obstacles)
        if planned_waypoint is not None:
            self.obstacle_avoidance_active = True
            self.current_obstacle_index = index
            self.current_waypoint = planned_waypoint
            return planned_waypoint, "obstacle_avoidance", index

        inflated_radius = obstacle.inflated_radius(self.robot_radius, self.safety_margin)
        waypoint_margin = self.waypoint_margin
        if self.waypoint_margin_boost_active:
            waypoint_margin += self.stall_waypoint_margin_boost
        candidates = compute_tangent_waypoints(
            pos,
            obstacle,
            inflated_radius,
            waypoint_margin,
        )
        previous_direction = self.tangent_direction if self.current_obstacle_index == index else None
        if self.waypoint_margin_boost_active and previous_direction in (-1, 1):
            previous_direction = -previous_direction
        waypoint, direction = choose_tangent_waypoint(
            pos,
            theta,
            goal,
            candidates,
            previous_direction=previous_direction,
        )

        self.obstacle_avoidance_active = True
        self.current_obstacle_index = index
        self.current_waypoint = waypoint
        self.tangent_direction = direction
        self.waypoint_margin_boost_active = False
        return waypoint, "obstacle_avoidance", index

    def _simple_bypass_waypoint(
        self,
        pos: np.ndarray,
        theta: float,
        goal: np.ndarray,
        obstacle: CircularObstacle,
        obstacles: tuple[CircularObstacle, ...],
    ) -> np.ndarray | None:
        """Choose one stable side waypoint around the blocking obstacle."""
        radius = (
            obstacle.inflated_radius(self.robot_radius, self.safety_margin)
            + self.waypoint_margin
            + self.planner_clearance
        )
        candidates = compute_side_bypass_waypoints(pos, goal, obstacle, radius)
        if self.tangent_direction in (-1, 1):
            ordered = (
                (self.tangent_direction, candidates[0 if self.tangent_direction == 1 else 1]),
                (-self.tangent_direction, candidates[1 if self.tangent_direction == 1 else 0]),
            )
        else:
            ordered = ((1, candidates[0]), (-1, candidates[1]))

        best: tuple[float, int, np.ndarray] | None = None
        for direction, waypoint in ordered:
            if not self._point_is_clear(waypoint, obstacles):
                continue
            if not self._segment_is_clear(pos, waypoint, obstacles):
                continue

            heading = float(np.arctan2(waypoint[1] - pos[1], waypoint[0] - pos[0]))
            heading_change = abs(float(wrap_to_pi(heading - theta)))
            path_cost = float(np.linalg.norm(waypoint - pos) + np.linalg.norm(goal - waypoint))
            cost = path_cost + 0.25 * heading_change
            if best is None or cost < best[0]:
                best = (cost, direction, waypoint)

        if best is None:
            return None

        self.tangent_direction = best[1]
        return best[2].copy()

    def _visibility_graph_waypoint(
        self,
        pos: np.ndarray,
        goal: np.ndarray,
        obstacles: tuple[CircularObstacle, ...],
    ) -> np.ndarray | None:
        """Plan a one-step waypoint using obstacle-boundary visibility nodes."""
        nodes = [pos.copy(), goal.copy()]
        for obstacle in obstacles:
            radius = (
                obstacle.inflated_radius(self.robot_radius, self.safety_margin)
                + self.waypoint_margin
                + self.planner_clearance
            )
            for i in range(self.visibility_samples_per_obstacle):
                angle = 2.0 * np.pi * i / self.visibility_samples_per_obstacle
                candidate = obstacle.center + radius * np.array([np.cos(angle), np.sin(angle)])
                if self._point_is_clear(candidate, obstacles):
                    nodes.append(candidate)

        count = len(nodes)
        distances = [np.inf] * count
        previous: list[int | None] = [None] * count
        visited = [False] * count
        distances[0] = 0.0

        for _ in range(count):
            current = None
            best_distance = np.inf
            for i in range(count):
                if not visited[i] and distances[i] < best_distance:
                    current = i
                    best_distance = distances[i]
            if current is None or current == 1:
                break

            visited[current] = True
            for neighbor in range(count):
                if neighbor == current or visited[neighbor]:
                    continue
                if not self._segment_is_clear(nodes[current], nodes[neighbor], obstacles):
                    continue
                edge_cost = float(np.linalg.norm(nodes[neighbor] - nodes[current]))
                new_distance = distances[current] + edge_cost
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = current

        if not np.isfinite(distances[1]):
            return None

        path = [1]
        current = previous[1]
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()
        if len(path) < 2:
            return None
        return nodes[path[1]].copy()

    def _point_is_clear(
        self,
        point: np.ndarray,
        obstacles: tuple[CircularObstacle, ...],
    ) -> bool:
        """Check that a candidate waypoint is outside every inflated obstacle."""
        for obstacle in obstacles:
            radius = (
                obstacle.inflated_radius(self.robot_radius, self.safety_margin)
                + self.planner_clearance
            )
            if float(np.linalg.norm(point - obstacle.center)) <= radius:
                return False
        return True

    def _segment_is_clear(
        self,
        start: np.ndarray,
        end: np.ndarray,
        obstacles: tuple[CircularObstacle, ...],
    ) -> bool:
        """Check that a segment avoids all inflated obstacles."""
        for obstacle in obstacles:
            radius = (
                obstacle.inflated_radius(self.robot_radius, self.safety_margin)
                + self.planner_clearance
            )
            if segment_circle_intersection(start, end, obstacle.center, radius):
                return False
        return True

    def _nearest_blocking_obstacle(
        self,
        pos: np.ndarray,
        goal: np.ndarray,
        obstacles: tuple[CircularObstacle, ...],
        extra_radius: float = 0.0,
        include_activation_zone: bool = True,
    ) -> tuple[int, CircularObstacle] | None:
        """Find the nearest obstacle blocking the direct path to the goal."""
        best: tuple[float, int, CircularObstacle] | None = None
        for index, obstacle in enumerate(obstacles):
            inflated_radius = obstacle.inflated_radius(self.robot_radius, self.safety_margin)
            path_radius = inflated_radius + extra_radius
            blocks_path = segment_circle_intersection(pos, goal, obstacle.center, path_radius)
            distance_to_boundary = float(np.linalg.norm(pos - obstacle.center) - inflated_radius)
            activated = (
                include_activation_zone
                and distance_to_boundary <= self.obstacle_activation_distance
            )
            if not blocks_path and not activated:
                continue

            distance_to_center = float(np.linalg.norm(pos - obstacle.center))
            if best is None or distance_to_center < best[0]:
                best = (distance_to_center, index, obstacle)

        if best is None:
            return None
        return best[1], best[2]

    def _unsafe_obstacle(
        self,
        pos: np.ndarray,
        obstacles: tuple[CircularObstacle, ...],
    ) -> tuple[int, CircularObstacle] | None:
        """Return an obstacle when the robot is inside its inflated radius."""
        for index, obstacle in enumerate(obstacles):
            inflated_radius = obstacle.inflated_radius(self.robot_radius, self.safety_margin)
            if float(np.linalg.norm(pos - obstacle.center)) < inflated_radius:
                return index, obstacle
        return None

    def _unsafe_recovery(
        self,
        pos: np.ndarray,
        theta: float,
        obstacle: CircularObstacle,
        obstacle_index: int,
    ) -> ControlDiagnostics:
        """Rotate away when numerical issues put the robot inside an obstacle."""
        away = pos - obstacle.center
        if float(np.linalg.norm(away)) <= 1e-9:
            desired_heading = theta + np.pi
        else:
            desired_heading = float(np.arctan2(away[1], away[0]))
        alpha = float(wrap_to_pi(desired_heading - theta))
        omega = float(self.k_alpha * np.sin(alpha))
        v = 0.0
        if abs(alpha) <= self.max_forward_heading_error:
            v = self.min_avoidance_speed
        elif abs(omega) < self.unsafe_turn_rate:
            omega = self.unsafe_turn_rate if alpha >= 0.0 else -self.unsafe_turn_rate

        u_l, u_r = unicycle_to_tracks(v, omega, self.b)
        if self.u_max is not None:
            u_l = float(np.clip(u_l, -self.u_max, self.u_max))
            u_r = float(np.clip(u_r, -self.u_max, self.u_max))
            v, omega = tracks_to_unicycle(u_l, u_r, self.b)

        return ControlDiagnostics(
            mode="unsafe_recovery",
            current_target=(float(pos[0]), float(pos[1])),
            blocking_obstacle_index=obstacle_index,
            rho=0.0,
            alpha=alpha,
            v=float(v),
            omega=float(omega),
            u_l=float(u_l),
            u_r=float(u_r),
            V=self.lyapunov_c * (1.0 - float(np.cos(alpha))),
            avoidance_active=True,
        )

    def _dodge_check(
        self,
        pos: np.ndarray,
        theta: float,
        moving_obstacles: tuple[MovingObstacle, ...],
    ) -> ControlDiagnostics | None:
        """Return a dodge command when a projectile will hit the robot.

        Strategy: ANGULAR BIAS INJECTION
          The robot does NOT stop to aim at an escape point.  Instead:
          1. The most imminent threat is identified (smallest t_cpa among bullets
             that will actually hit).
          2. A steering correction Δω is added on top of the normal Lyapunov output
             toward the robot's current navigation target (goal or active waypoint).
          3. v is preserved — the robot keeps moving forward at its natural speed.

          This prevents the "stop and spin" failure mode where high heading error
          to a lateral escape point drives v ≈ 0 and ω ≫ 0.

        Δω magnitude is proportional to threat severity (how deep inside the
        collision radius the CPA point falls) and scales with k_alpha.
        Direction is chosen so the robot steers toward the side that moves it
        away from the bullet's line of travel, preferring the side closer to goal.
        """
        goal_xy = np.asarray(self.goal, dtype=float)

        # ── Step 1: find the single most imminent real threat ───────────────
        best: tuple | None = None
        best_t_cpa = float("inf")

        for obs in moving_obstacles:
            v_b = obs.velocity
            speed = float(np.linalg.norm(v_b))
            if speed < 1e-9:
                continue

            p_b = obs.position
            coll_r = (obs.radius + self.robot_radius) * self.dodge_danger_factor

            # Immediate proximity gate: bullet already inside danger radius.
            # This catches bullets that slipped past the lookahead window (e.g.
            # because they were just outside range one step ago and then closed
            # faster than 1 timestep).  Skip the dot-product / CPA checks — the
            # bullet is already on top of us, treat t_cpa = 0.
            current_dist = float(np.linalg.norm(pos - p_b))
            if current_dist < coll_r:
                if 0.0 < best_t_cpa:   # always beats any CPA-based candidate
                    best_t_cpa = 0.0
                    best = (v_b, speed, p_b, 0.0, p_b.copy(), current_dist, coll_r)
                continue

            # Bullet must be moving toward the robot
            if float(np.dot(v_b, pos - p_b)) <= 0.0:
                continue

            t_cpa = float(np.dot(pos - p_b, v_b)) / float(np.dot(v_b, v_b))
            t_cpa = max(0.0, t_cpa)
            if t_cpa > self.dodge_lookahead:
                continue

            cpa = p_b + t_cpa * v_b
            miss_dist = float(np.linalg.norm(cpa - pos))

            if miss_dist > coll_r:
                continue  # will miss — no action needed

            # Track the most imminent (smallest t_cpa) threat
            if t_cpa < best_t_cpa:
                best_t_cpa = t_cpa
                best = (v_b, speed, p_b, t_cpa, cpa, miss_dist, coll_r)

        if best is None:
            return None

        v_b, speed, p_b, t_cpa, cpa, miss_dist, coll_r = best

        # ── Step 2: determine which side of the bullet's line to move to ────
        v_hat = v_b / speed
        perp_pos = np.array([-v_b[1], v_b[0]], dtype=float) / speed  # 90° CCW

        # Perpendicular component from bullet origin to robot — tells us which
        # side the robot is currently on relative to the bullet's line.
        proj_to_robot = pos - p_b
        along = float(np.dot(proj_to_robot, v_hat))
        perp_vec = proj_to_robot - along * v_hat   # perpendicular offset vector
        h = float(np.linalg.norm(perp_vec))

        if h > 1e-4:
            # Continue moving in the same lateral direction (away from bullet line)
            lateral_dir = perp_vec / h
        else:
            # Robot exactly on bullet line — pick side closer to goal
            side_a = pos + 0.5 * perp_pos
            side_b = pos - 0.5 * perp_pos
            if float(np.linalg.norm(side_a - goal_xy)) <= \
               float(np.linalg.norm(side_b - goal_xy)):
                lateral_dir = perp_pos
            else:
                lateral_dir = -perp_pos

        # ── Step 3: compute Δω to steer in lateral_dir ──────────────────────
        # Positive ω = CCW = robot turns left.  "Left" in robot frame:
        left_dir = np.array([-np.sin(theta), np.cos(theta)])
        omega_sign = 1.0 if float(np.dot(lateral_dir, left_dir)) >= 0.0 else -1.0

        # Magnitude: proportional to threat severity, but intentionally mild.
        # At lookahead 1.0s and v≥0.7 m/s, Δω=1.5 rad/s gives lateral arc
        # ≈ 0.5·v·Δω·t² ≈ 0.5·0.7·1.5·1² ≈ 0.5 m — just enough to clear the
        # hitbox sum (~0.53 m).  Using self.k_alpha (=3.0) here would create
        # R=v/ω≈0.12 m circles and make the robot spin instead of arc.
        threat_fraction = max(0.0, (coll_r - miss_dist) / coll_r)
        _DODGE_W_GAIN = 2.0   # rad/s at max threat — gentle, predictable arc
        omega_correction = omega_sign * _DODGE_W_GAIN * (0.5 + threat_fraction)
        # Hard cap: never exceed half the actuator limit (leaves room for v)
        if self.u_max is not None:
            max_w = self.u_max / self.b   # half of actuator-max angular rate
            omega_correction = float(np.clip(omega_correction, -max_w, max_w))

        # ── Step 4: compute normal Lyapunov output toward the nav target ────
        # If obstacle avoidance is active, keep targeting the current waypoint
        # so the robot does not steer into obstacles.  Otherwise target goal.
        # Either way we enforce a minimum forward speed below so the robot never
        # stalls during a dodge (waypoints behind the robot produce v<0 which
        # would make us a stationary target — the min-v clamp handles this).
        if self.obstacle_avoidance_active and self.current_waypoint is not None:
            nav_target = np.asarray(self.current_waypoint, dtype=float)
        else:
            nav_target = goal_xy

        normal_diag = self._lyapunov_diagnostics(
            state=(float(pos[0]), float(pos[1]), theta),
            target=nav_target,
            mode="dodge",
            blocking_obstacle_index=None,
        )

        # ── Step 5: inject Δω, enforce minimum forward speed, re-clip ────────
        new_omega = normal_diag.omega + omega_correction
        # Clamp v ≥ _MIN_DODGE_V: a moving robot is far harder to hit than a
        # stationary one.  This is the key fix — previously the avoidance-motion
        # guard or a large heading error could drive v to zero during dodge.
        _MIN_DODGE_V = 1.0   # m/s — R = v/ω = 1.0/2.0 = 0.5 m arc, clears 0.53 m hitbox
        base_v = max(normal_diag.v, _MIN_DODGE_V)
        u_l, u_r = unicycle_to_tracks(base_v, new_omega, self.b)
        if self.u_max is not None:
            u_l = float(np.clip(u_l, -self.u_max, self.u_max))
            u_r = float(np.clip(u_r, -self.u_max, self.u_max))
            new_v, new_omega = tracks_to_unicycle(u_l, u_r, self.b)
        else:
            new_v = base_v
            u_l, u_r = unicycle_to_tracks(new_v, new_omega, self.b)

        return ControlDiagnostics(
            mode="dodge",
            current_target=normal_diag.current_target,
            blocking_obstacle_index=None,
            rho=normal_diag.rho,
            alpha=normal_diag.alpha,
            v=new_v,
            omega=new_omega,
            u_l=u_l,
            u_r=u_r,
            V=normal_diag.V,
            avoidance_active=self.obstacle_avoidance_active,
        )

    def _clear_avoidance(self) -> None:
        """Reset obstacle-avoidance supervisor state."""
        self.obstacle_avoidance_active = False
        self.current_waypoint = None
        self.current_obstacle_index = None
        self.tangent_direction = None
        self.last_position = None
        self.last_goal_distance = None
        self.stall_counter = 0
        self.waypoint_margin_boost_active = False

    def _update_stall_state(self, pos: np.ndarray, goal: np.ndarray) -> bool:
        """Detect lack of motion or lack of progress to the final goal."""
        goal_distance = float(np.linalg.norm(goal - pos))
        if not self.obstacle_avoidance_active:
            self.last_position = pos.copy()
            self.last_goal_distance = goal_distance
            self.stall_counter = 0
            return False

        if self.last_position is None or self.last_goal_distance is None:
            self.last_position = pos.copy()
            self.last_goal_distance = goal_distance
            self.stall_counter = 0
            return False

        moved = float(np.linalg.norm(pos - self.last_position))
        progress = self.last_goal_distance - goal_distance
        if moved < self.stall_position_epsilon or progress < self.stall_goal_progress_epsilon:
            self.stall_counter += 1
        else:
            self.stall_counter = 0

        self.last_position = pos.copy()
        self.last_goal_distance = goal_distance
        if self.stall_counter < self.stall_steps:
            return False

        self.stall_counter = 0
        if self.tangent_direction in (-1, 1):
            self.tangent_direction = -self.tangent_direction
        else:
            self.tangent_direction = 1
        self.waypoint_margin_boost_active = True
        return True

    def _avoidance_motion_guard(
        self,
        v: float,
        omega: float,
        alpha: float,
        stalled: bool,
    ) -> tuple[float, float]:
        """Keep the robot moving during obstacle-avoidance recovery."""
        if abs(alpha) > self.max_forward_heading_error:
            return 0.0, omega

        min_speed = self.min_avoidance_speed
        if stalled:
            min_speed *= 1.5

        if v < min_speed:
            v = min_speed
        return v, omega


def unicycle_to_tracks(v: float, omega: float, b: float) -> tuple[float, float]:
    """Map unicycle command (v, omega) to track velocities (u_L, u_R)."""
    if b <= 0.0:
        raise ValueError("b must be positive")
    return float(v - 0.5 * b * omega), float(v + 0.5 * b * omega)


def tracks_to_unicycle(u_l: float, u_r: float, b: float) -> tuple[float, float]:
    """Map track velocities (u_L, u_R) to unicycle command (v, omega)."""
    if b <= 0.0:
        raise ValueError("b must be positive")
    return float(0.5 * (u_r + u_l)), float((u_r - u_l) / b)


def compute_control(
    state: object,
    goal: object,
    obstacles: Iterable[object] | None = None,
    params: Mapping[str, float] | None = None,
) -> tuple[float, float, ControlDiagnostics]:
    """Compute saturated track commands and debug data."""
    kwargs = dict(params or {})
    controller = LyapunovTrackedRobotController(
        goal=goal,
        obstacles=tuple(obstacles or ()),
        **kwargs,
    )
    return controller.get_control_with_debug(state)
