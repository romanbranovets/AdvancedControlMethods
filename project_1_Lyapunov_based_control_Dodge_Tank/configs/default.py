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
    goal_tolerance: float = 0.35


@dataclass(frozen=True, slots=True)
class NoiseConfig:
    """Additive Gaussian measurement noise on the robot state.

    Models sensor imperfections (GPS jitter, IMU drift).
    Noise is applied to the observed state before passing it to the controller;
    the simulator always integrates the true (noiseless) state.
    """

    enabled: bool = True
    position_std: float = 0.03   # σ_pos — std of x and y noise, m
    heading_std: float = 0.015   # σ_hdg — std of θ noise, rad
    seed: int | None = None      # None → different sequence each run


@dataclass(frozen=True, slots=True)
class CannonConfig:
    """Stationary cannon that fires projectiles as a Poisson process.

    Inter-arrival times are i.i.d. Exp(1/mean_fire_interval).
    Each shot is aimed at the robot with Gaussian angular noise.
    """

    enabled: bool = True
    x: float = 9.0                    # cannon x position, m
    y: float = -6.0                   # cannon y position, m
    mean_fire_interval: float = 1.5   # mean seconds between shots (Poisson rate λ = 1/1.5)
    projectile_speed: float = 7.5     # muzzle speed, m/s
    projectile_radius: float = 0.18   # collision radius of each shell, m
    angular_spread_std: float = 0.10  # Gaussian aim noise std, rad (~6°)
    max_projectile_age: float = 10.0  # seconds until a shell is removed
    seed: int | None = None           # None → different sequence each run
    # Dodge parameters (controller reacts to incoming projectiles)
    dodge_lookahead: float = 1.2        # seconds — react early enough to steer clear
    dodge_danger_factor: float = 1.4   # react to near-misses within 40% buffer


@dataclass(frozen=True, slots=True)
class ControllerConfig:
    """Default Lyapunov tracked-robot controller settings."""

    k_rho: float = 0.8
    k_alpha: float = 3.0
    b: float = 0.52
    u_max: float | None = 2.0
    eps_goal: float = 0.35
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
    obstacle_count_range: tuple[int, int] = (4, 7)
    obstacle_radius_range: tuple[float, float] = (0.45, 1.1)
    x_range: tuple[float, float] = (-6.0, 5.0)
    y_range: tuple[float, float] = (-4.0, 6.0)
    start_x_range: tuple[float, float] = (-8.0, -3.0)
    start_y_range: tuple[float, float] = (-5.0, 5.0)
    start_theta_range: tuple[float, float] = (-3.141592653589793, 3.141592653589793)
    min_start_goal_distance: float = 7.0
    min_clearance: float = 0.8
    max_attempts: int = 500
    validate_scenario: bool = True
    validation_attempts: int = 30
    validation_steps: int = 2000
    validation_goal_tolerance: float = 0.4


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Top-level application configuration."""

    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    random: RandomScenarioConfig = field(default_factory=RandomScenarioConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    cannon: CannonConfig = field(default_factory=CannonConfig)
    goal: tuple[float, float] = (6.0, 4.0)
    obstacles: tuple[tuple[float, float, float], ...] = ()
    render_every: int = 10


DEFAULT_CONFIG = AppConfig()
