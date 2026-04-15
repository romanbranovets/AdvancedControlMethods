from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Default simulation settings."""

    num_steps: int = 2000
    dt: float = 0.05
    initial_state: tuple[float, float, float] = (0.0, 0.0, 0.0)
    command_mode: str = "tracks"
    stop_at_goal: bool = True
    goal_tolerance: float = 0.05


@dataclass(frozen=True, slots=True)
class ControllerConfig:
    """Default Lyapunov tracked-robot controller settings."""

    k_rho: float = 0.8
    k_alpha: float = 4.0
    b: float = 0.52
    u_max: float | None = 2.0
    eps_goal: float = 0.05
    lyapunov_c: float = 1.0
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


@dataclass(frozen=True, slots=True)
class RandomScenarioConfig:
    """Random start and obstacle generation settings."""

    enabled: bool = True
    seed: int | None = None
    obstacle_count_range: tuple[int, int] = (2, 5)
    obstacle_radius_range: tuple[float, float] = (0.35, 0.9)
    x_range: tuple[float, float] = (-4.0, 4.0)
    y_range: tuple[float, float] = (-3.0, 4.0)
    start_x_range: tuple[float, float] = (-4.0, 0.5)
    start_y_range: tuple[float, float] = (-3.0, 3.0)
    start_theta_range: tuple[float, float] = (-3.141592653589793, 3.141592653589793)
    min_start_goal_distance: float = 2.0
    min_clearance: float = 1.0
    max_attempts: int = 500
    validate_scenario: bool = True
    validation_attempts: int = 30
    validation_steps: int = 1000
    validation_goal_tolerance: float = 0.1


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Top-level application configuration."""

    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    random: RandomScenarioConfig = field(default_factory=RandomScenarioConfig)
    goal: tuple[float, float] = (5.0, 3.0)
    obstacles: tuple[tuple[float, float, float], ...] = ()
    render_every: int = 10


DEFAULT_CONFIG = AppConfig()
