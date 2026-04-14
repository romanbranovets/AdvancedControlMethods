from __future__ import annotations

import argparse
from typing import Iterable

import numpy as np

from configs import DEFAULT_CONFIG, CannonConfig, NoiseConfig, RandomScenarioConfig
from src.cannon import Cannon, Projectile
from src.controller import LyapunovTrackedRobotController, MovingObstacle
from src.simulation import TrackedRobotSim
from src.system import add_measurement_noise
from src.visualization import Visualizer

CFG = DEFAULT_CONFIG


def _sample_uniform_pair(
    rng: np.random.Generator,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> np.ndarray:
    return np.array(
        [
            rng.uniform(x_range[0], x_range[1]),
            rng.uniform(y_range[0], y_range[1]),
        ],
        dtype=float,
    )


def generate_random_scenario(
    goal: Iterable[float],
    config: RandomScenarioConfig = CFG.random,
    robot_radius: float = CFG.controller.robot_radius,
    safety_margin: float = CFG.controller.safety_margin,
) -> tuple[tuple[float, float, float], tuple[tuple[float, float, float], ...]]:
    """Generate a random initial pose and non-overlapping circular obstacles."""
    if config.obstacle_count_range[0] < 0 or config.obstacle_count_range[1] < config.obstacle_count_range[0]:
        raise ValueError("invalid obstacle_count_range")

    rng = np.random.default_rng(config.seed)
    goal_xy = np.asarray(goal, dtype=float).reshape(2)

    start_xy: np.ndarray | None = None
    for _ in range(config.max_attempts):
        candidate = _sample_uniform_pair(rng, config.start_x_range, config.start_y_range)
        if float(np.linalg.norm(candidate - goal_xy)) >= config.min_start_goal_distance:
            start_xy = candidate
            break
    if start_xy is None:
        raise RuntimeError("failed to sample a valid random start")

    theta = float(rng.uniform(config.start_theta_range[0], config.start_theta_range[1]))
    start = (float(start_xy[0]), float(start_xy[1]), theta)

    min_count, max_count = config.obstacle_count_range
    obstacle_count = int(rng.integers(min_count, max_count + 1))
    obstacles: list[tuple[float, float, float]] = []
    for _ in range(obstacle_count):
        for _attempt in range(config.max_attempts):
            center = _sample_uniform_pair(rng, config.x_range, config.y_range)
            radius = float(rng.uniform(config.obstacle_radius_range[0], config.obstacle_radius_range[1]))
            inflated = radius + robot_radius + safety_margin + config.min_clearance

            if float(np.linalg.norm(center - start_xy)) <= inflated:
                continue
            if float(np.linalg.norm(center - goal_xy)) <= inflated:
                continue

            overlaps = False
            for x_obs, y_obs, r_obs in obstacles:
                other_center = np.array([x_obs, y_obs], dtype=float)
                min_distance = radius + r_obs + robot_radius + safety_margin + config.min_clearance
                if float(np.linalg.norm(center - other_center)) <= min_distance:
                    overlaps = True
                    break
            if overlaps:
                continue

            obstacles.append((float(center[0]), float(center[1]), radius))
            break

    return start, tuple(obstacles)


def random_config_with_seed(config: RandomScenarioConfig, seed: int | None) -> RandomScenarioConfig:
    """Return a copy of a random config with a specific seed."""
    return RandomScenarioConfig(
        enabled=config.enabled,
        seed=seed,
        obstacle_count_range=config.obstacle_count_range,
        obstacle_radius_range=config.obstacle_radius_range,
        x_range=config.x_range,
        y_range=config.y_range,
        start_x_range=config.start_x_range,
        start_y_range=config.start_y_range,
        start_theta_range=config.start_theta_range,
        min_start_goal_distance=config.min_start_goal_distance,
        min_clearance=config.min_clearance,
        max_attempts=config.max_attempts,
        validate_scenario=config.validate_scenario,
        validation_attempts=config.validation_attempts,
        validation_steps=config.validation_steps,
        validation_goal_tolerance=config.validation_goal_tolerance,
    )


def make_random_config_for_args(args: argparse.Namespace) -> RandomScenarioConfig:
    """Build random scenario config using CLI seed override."""
    return RandomScenarioConfig(
        enabled=CFG.random.enabled,
        seed=args.seed,
        obstacle_count_range=CFG.random.obstacle_count_range,
        obstacle_radius_range=CFG.random.obstacle_radius_range,
        x_range=CFG.random.x_range,
        y_range=CFG.random.y_range,
        start_x_range=CFG.random.start_x_range,
        start_y_range=CFG.random.start_y_range,
        start_theta_range=CFG.random.start_theta_range,
        min_start_goal_distance=CFG.random.min_start_goal_distance,
        min_clearance=CFG.random.min_clearance,
        max_attempts=CFG.random.max_attempts,
        validate_scenario=CFG.random.validate_scenario,
        validation_attempts=CFG.random.validation_attempts,
        validation_steps=CFG.random.validation_steps,
        validation_goal_tolerance=CFG.random.validation_goal_tolerance,
    )


def scenario_reaches_goal(
    initial_state: Iterable[float],
    goal: Iterable[float],
    obstacles: Iterable[object],
    steps: int,
    tolerance: float,
) -> bool:
    """Dry-run a scenario and return True if the controller reaches the goal."""
    sim = run(
        num_steps=steps,
        initial_state=initial_state,
        goal=goal,
        obstacles=obstacles,
        render=False,
    )
    final_state = sim.pose
    goal_xy = np.asarray(goal, dtype=float).reshape(2)
    return float(np.linalg.norm(final_state[:2] - goal_xy)) <= tolerance


def generate_valid_random_scenario(
    goal: Iterable[float],
    config: RandomScenarioConfig,
) -> tuple[tuple[float, float, float], tuple[tuple[float, float, float], ...]]:
    """Generate a random scenario, rejecting cases where the controller stalls."""
    attempts = config.validation_attempts if config.validate_scenario else 1
    last_scenario: tuple[tuple[float, float, float], tuple[tuple[float, float, float], ...]] | None = None
    for attempt in range(attempts):
        seed = None if config.seed is None else config.seed + attempt
        scenario_config = random_config_with_seed(config, seed)
        scenario = generate_random_scenario(
            goal=goal,
            config=scenario_config,
            robot_radius=CFG.controller.robot_radius,
            safety_margin=CFG.controller.safety_margin,
        )
        last_scenario = scenario
        if not config.validate_scenario:
            return scenario
        if scenario_reaches_goal(
            initial_state=scenario[0],
            goal=goal,
            obstacles=scenario[1],
            steps=config.validation_steps,
            tolerance=config.validation_goal_tolerance,
        ):
            return scenario

    fallback_config = random_config_with_seed(config, config.seed)
    fallback_start, _ = generate_random_scenario(
        goal=goal,
        config=RandomScenarioConfig(
            enabled=fallback_config.enabled,
            seed=fallback_config.seed,
            obstacle_count_range=(0, 0),
            obstacle_radius_range=fallback_config.obstacle_radius_range,
            x_range=fallback_config.x_range,
            y_range=fallback_config.y_range,
            start_x_range=fallback_config.start_x_range,
            start_y_range=fallback_config.start_y_range,
            start_theta_range=fallback_config.start_theta_range,
            min_start_goal_distance=fallback_config.min_start_goal_distance,
            min_clearance=fallback_config.min_clearance,
            max_attempts=fallback_config.max_attempts,
            validate_scenario=False,
            validation_attempts=1,
            validation_steps=fallback_config.validation_steps,
            validation_goal_tolerance=fallback_config.validation_goal_tolerance,
        ),
        robot_radius=CFG.controller.robot_radius,
        safety_margin=CFG.controller.safety_margin,
    )
    return fallback_start, ()


def run(
    num_steps: int = CFG.simulation.num_steps,
    dt: float = CFG.simulation.dt,
    initial_state: Iterable[float] = CFG.simulation.initial_state,
    goal: Iterable[float] = CFG.goal,
    obstacles: Iterable[object] = CFG.obstacles,
    k_rho: float = CFG.controller.k_rho,
    k_alpha: float = CFG.controller.k_alpha,
    b: float = CFG.controller.b,
    u_max: float | None = CFG.controller.u_max,
    eps_goal: float = CFG.controller.eps_goal,
    lyapunov_c: float = CFG.controller.lyapunov_c,
    robot_radius: float = CFG.controller.robot_radius,
    safety_margin: float = CFG.controller.safety_margin,
    obstacle_activation_distance: float = CFG.controller.obstacle_activation_distance,
    obstacle_clearance_hysteresis: float = CFG.controller.obstacle_clearance_hysteresis,
    waypoint_margin: float = CFG.controller.waypoint_margin,
    unsafe_turn_rate: float = CFG.controller.unsafe_turn_rate,
    min_avoidance_speed: float = CFG.controller.min_avoidance_speed,
    max_forward_heading_error: float = CFG.controller.max_forward_heading_error,
    stall_position_epsilon: float = CFG.controller.stall_position_epsilon,
    stall_goal_progress_epsilon: float = CFG.controller.stall_goal_progress_epsilon,
    stall_steps: int = CFG.controller.stall_steps,
    stall_waypoint_margin_boost: float = CFG.controller.stall_waypoint_margin_boost,
    visibility_samples_per_obstacle: int = CFG.controller.visibility_samples_per_obstacle,
    planner_clearance: float = CFG.controller.planner_clearance,
    waypoint_reached_radius: float = CFG.controller.waypoint_reached_radius,
    command_mode: str = CFG.simulation.command_mode,
    stop_at_goal: bool = CFG.simulation.stop_at_goal,
    goal_tolerance: float = CFG.simulation.goal_tolerance,
    noise_config: NoiseConfig = CFG.noise,
    cannon_config: CannonConfig = CFG.cannon,
    render: bool = False,
    render_every: int = CFG.render_every,
) -> TrackedRobotSim:
    sim = TrackedRobotSim(dt=dt, max_track_speed=u_max)
    ctrl = LyapunovTrackedRobotController(
        goal=goal,
        k_rho=k_rho,
        k_alpha=k_alpha,
        b=b,
        u_max=u_max,
        eps_goal=eps_goal,
        lyapunov_c=lyapunov_c,
        obstacles=obstacles,
        robot_radius=robot_radius,
        safety_margin=safety_margin,
        obstacle_activation_distance=obstacle_activation_distance,
        obstacle_clearance_hysteresis=obstacle_clearance_hysteresis,
        waypoint_margin=waypoint_margin,
        unsafe_turn_rate=unsafe_turn_rate,
        min_avoidance_speed=min_avoidance_speed,
        max_forward_heading_error=max_forward_heading_error,
        stall_position_epsilon=stall_position_epsilon,
        stall_goal_progress_epsilon=stall_goal_progress_epsilon,
        stall_steps=stall_steps,
        stall_waypoint_margin_boost=stall_waypoint_margin_boost,
        visibility_samples_per_obstacle=visibility_samples_per_obstacle,
        planner_clearance=planner_clearance,
        waypoint_reached_radius=waypoint_reached_radius,
        dodge_lookahead=cannon_config.dodge_lookahead,
        dodge_escape_distance=cannon_config.dodge_escape_distance,
        dodge_danger_factor=cannon_config.dodge_danger_factor,
        output_mode=command_mode,
    )

    # --- Cannon setup ---
    cannon: Cannon | None = None
    if cannon_config.enabled:
        cannon = Cannon(
            x=cannon_config.x,
            y=cannon_config.y,
            mean_fire_interval=cannon_config.mean_fire_interval,
            projectile_speed=cannon_config.projectile_speed,
            projectile_radius=cannon_config.projectile_radius,
            angular_spread_std=cannon_config.angular_spread_std,
            max_projectile_age=cannon_config.max_projectile_age,
            rng=np.random.default_rng(cannon_config.seed),
        )
        print(
            f"[Cannon] enabled at ({cannon_config.x}, {cannon_config.y}) | "
            f"mean interval={cannon_config.mean_fire_interval}s | "
            f"speed={cannon_config.projectile_speed} m/s | "
            f"seed={cannon_config.seed}"
        )

    # --- Noise RNG and EMA filter ---
    # EMA (exponential moving average) smooths noisy measurements before they
    # reach the controller, preventing high-frequency chatter near the goal.
    # α_ema close to 1 → fast tracking; close to 0 → heavy smoothing.
    _EMA_ALPHA = 0.55
    noise_rng: np.random.Generator | None = None
    if noise_config.enabled:
        noise_rng = np.random.default_rng(noise_config.seed)

    active_projectiles: list[Projectile] = []
    state = sim.reset(initial_state)
    goal_xy = np.asarray(goal, dtype=float).reshape(2)
    filtered_state: np.ndarray = state.copy()   # EMA state; initialised to true state

    for _ in range(num_steps):
        # 1. Cannon fires (Poisson process)
        if cannon is not None:
            new_proj = cannon.update(dt, sim.t, state[:2])
            if new_proj is not None:
                active_projectiles.append(new_proj)
                print(
                    f"  [Cannon] Shot #{cannon.shot_count} at t={sim.t:.2f}s | "
                    f"angle={np.degrees(cannon.fire_log[-1]['fire_angle_rad']):.1f}° | "
                    f"noise={np.degrees(cannon.fire_log[-1]['noise_angle_rad']):.1f}°"
                )

        # 2. Advance projectiles
        for proj in active_projectiles:
            proj.step(dt, sim.xlim, sim.ylim, cannon_config.max_projectile_age)
        active_projectiles = [p for p in active_projectiles if p.alive]

        # 3. Record projectile snapshot for this step
        sim.record_projectile_snapshot(
            [(p.x, p.y, p.radius) for p in active_projectiles]
        )

        # 4. Apply measurement noise then EMA low-pass filter.
        #    The simulator always integrates the *true* state; only the
        #    controller input is noisy + filtered.
        if noise_rng is not None:
            noisy_state = add_measurement_noise(
                state, noise_config.position_std, noise_config.heading_std, noise_rng
            )
        else:
            noisy_state = state

        # EMA update (linear blend; angle wrap applied to θ component)
        filtered_state = _EMA_ALPHA * noisy_state + (1.0 - _EMA_ALPHA) * filtered_state
        filtered_state[2] = float(
            (filtered_state[2] + np.pi) % (2.0 * np.pi) - np.pi
        )
        observed_state = filtered_state

        # 5. Build moving-obstacle list from active projectiles
        moving_obs = [
            MovingObstacle(p.x, p.y, p.vx, p.vy, p.radius)
            for p in active_projectiles
        ]

        # 6. Compute control
        u_l, u_r, debug = ctrl.get_control_with_debug(
            observed_state, moving_obstacles=moving_obs
        )
        action = np.array([u_l, u_r], dtype=float)
        if command_mode == "vw":
            action = np.array([debug.v, debug.omega], dtype=float)
        sim.record_controller_debug(debug.current_target, debug.mode)

        # 7. Advance robot (true state)
        state = sim.step(action, command_mode=command_mode)

        if stop_at_goal and float(np.linalg.norm(state[:2] - goal_xy)) <= goal_tolerance:
            break

    if cannon is not None:
        print(
            f"[Cannon] Total shots fired: {cannon.shot_count} | "
            f"Projectiles still alive at end: {len(active_projectiles)}"
        )

    if render:
        _ = render_every
        Visualizer(
            sim=sim,
            goal=goal,
            obstacles=tuple(obstacles),
            cannon_pos=(cannon_config.x, cannon_config.y) if cannon_config.enabled else None,
        ).render(realtime=True, repeat=False)

    return sim


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple tracked robot simulation runner")
    parser.add_argument(
        "--num-steps",
        type=int,
        default=CFG.simulation.num_steps,
        help="Number of simulation steps",
    )
    parser.add_argument("--dt", type=float, default=CFG.simulation.dt, help="Simulation time step")
    parser.add_argument("--x0", type=float, default=CFG.simulation.initial_state[0], help="Initial x")
    parser.add_argument("--y0", type=float, default=CFG.simulation.initial_state[1], help="Initial y")
    parser.add_argument("--th0", type=float, default=CFG.simulation.initial_state[2], help="Initial heading")
    parser.add_argument("--goal-x", type=float, default=CFG.goal[0], help="Target x")
    parser.add_argument("--goal-y", type=float, default=CFG.goal[1], help="Target y")
    parser.add_argument("--k-rho", type=float, default=CFG.controller.k_rho, help="Position gain")
    parser.add_argument("--k-alpha", type=float, default=CFG.controller.k_alpha, help="Heading gain")
    parser.add_argument("--b", type=float, default=CFG.controller.b, help="Distance between track centers")
    parser.add_argument("--u-max", type=float, default=CFG.controller.u_max, help="Track velocity saturation")
    parser.add_argument("--eps-goal", type=float, default=CFG.controller.eps_goal, help="Goal tolerance")
    parser.add_argument(
        "--command-mode",
        choices=("tracks", "vw"),
        default=CFG.simulation.command_mode,
        help="Command format passed to simulator step",
    )
    parser.add_argument("--render", action="store_true", help="Render trajectory during simulation")
    parser.add_argument("--render-every", type=int, default=CFG.render_every, help="Render every N steps")
    parser.add_argument("--animate", action="store_true", help="Show final animation after run")
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of random scenarios to run; use 0 for endless animation loop",
    )
    parser.add_argument(
        "--no-random",
        action="store_true",
        help="Use fixed initial state and obstacles from the config/CLI",
    )
    parser.add_argument("--seed", type=int, default=CFG.random.seed, help="Random scenario seed")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    goal = (args.goal_x, args.goal_y)
    episode = 0

    while args.episodes == 0 or episode < args.episodes:
        if CFG.random.enabled and not args.no_random:
            base_random_config = make_random_config_for_args(args)
            seed = None if base_random_config.seed is None else base_random_config.seed + episode
            random_config = random_config_with_seed(base_random_config, seed)
            initial_state, obstacles = generate_valid_random_scenario(
                goal=goal,
                config=random_config,
            )
        else:
            initial_state = (args.x0, args.y0, args.th0)
            obstacles = CFG.obstacles

        sim = run(
            num_steps=args.num_steps,
            dt=args.dt,
            initial_state=initial_state,
            goal=goal,
            obstacles=obstacles,
            k_rho=args.k_rho,
            k_alpha=args.k_alpha,
            b=args.b,
            u_max=args.u_max,
            eps_goal=args.eps_goal,
            lyapunov_c=CFG.controller.lyapunov_c,
            robot_radius=CFG.controller.robot_radius,
            safety_margin=CFG.controller.safety_margin,
            obstacle_activation_distance=CFG.controller.obstacle_activation_distance,
            obstacle_clearance_hysteresis=CFG.controller.obstacle_clearance_hysteresis,
            waypoint_margin=CFG.controller.waypoint_margin,
            unsafe_turn_rate=CFG.controller.unsafe_turn_rate,
            min_avoidance_speed=CFG.controller.min_avoidance_speed,
            max_forward_heading_error=CFG.controller.max_forward_heading_error,
            stall_position_epsilon=CFG.controller.stall_position_epsilon,
            stall_goal_progress_epsilon=CFG.controller.stall_goal_progress_epsilon,
            stall_steps=CFG.controller.stall_steps,
            stall_waypoint_margin_boost=CFG.controller.stall_waypoint_margin_boost,
            visibility_samples_per_obstacle=CFG.controller.visibility_samples_per_obstacle,
            planner_clearance=CFG.controller.planner_clearance,
            waypoint_reached_radius=CFG.controller.waypoint_reached_radius,
            command_mode=args.command_mode,
            stop_at_goal=CFG.simulation.stop_at_goal,
            goal_tolerance=CFG.simulation.goal_tolerance,
            noise_config=CFG.noise,
            cannon_config=CFG.cannon,
            render=args.render,
            render_every=args.render_every,
        )

        final_state = sim.pose
        final_error = np.hypot(final_state[0] - args.goal_x, final_state[1] - args.goal_y)
        print(
            f"Episode {episode + 1}:",
            f"start={tuple(round(v, 3) for v in initial_state)}",
            f"obstacles={obstacles}",
        )
        print(
            "Final state:",
            f"x={final_state[0]:.3f}",
            f"y={final_state[1]:.3f}",
            f"theta={final_state[2]:.3f}",
            f"t={sim.t:.3f}",
            f"goal_error={final_error:.3f}",
        )

        if args.animate:
            Visualizer(
                sim=sim,
                goal=goal,
                obstacles=obstacles,
                cannon_pos=(CFG.cannon.x, CFG.cannon.y) if CFG.cannon.enabled else None,
            ).render(
                realtime=True,
                repeat=False,
                close_on_finish=args.episodes != 1,
            )

        episode += 1
        if not args.animate and args.episodes == 1:
            break


if __name__ == "__main__":
    main()
