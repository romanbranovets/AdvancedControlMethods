from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.simulation import TrackedRobotSim
from src.visualization import Visualizer


@dataclass(slots=True)
class Controller:
    v_l: float = 0.6
    v_r: float = 0.8

    def __call__(self, t: float, state: np.ndarray) -> np.ndarray:
        _ = (t, state)
        return np.array([self.v_l, self.v_r], dtype=float)


def run(
    num_steps: int = 200,
    dt: float = 0.05,
    initial_state: Iterable[float] = (0.0, 0.0, 0.0),
    v_l: float = 0.6,
    v_r: float = 0.8,
    command_mode: str = "tracks",
    render: bool = False,
    render_every: int = 1,
) -> TrackedRobotSim:
    sim = TrackedRobotSim(dt=dt)
    ctrl = Controller(v_l=v_l, v_r=v_r)

    state = sim.reset(initial_state)
    for _ in range(num_steps):
        t = sim.t
        action = ctrl(t, state)
        state = sim.step(action, command_mode=command_mode)

    if render:
        _ = render_every
        Visualizer(sim=sim).render(realtime=True, repeat=False)

    return sim


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple tracked robot simulation runner")
    parser.add_argument("--num-steps", type=int, default=200, help="Number of simulation steps")
    parser.add_argument("--dt", type=float, default=0.05, help="Simulation time step")
    parser.add_argument("--x0", type=float, default=0.0, help="Initial x")
    parser.add_argument("--y0", type=float, default=0.0, help="Initial y")
    parser.add_argument("--th0", type=float, default=0.0, help="Initial heading")
    parser.add_argument("--v-l", type=float, default=0.6, help="Left track velocity")
    parser.add_argument("--v-r", type=float, default=0.8, help="Right track velocity")
    parser.add_argument(
        "--command-mode",
        choices=("tracks", "vw"),
        default="tracks",
        help="Command format passed to simulator step",
    )
    parser.add_argument("--render", action="store_true", help="Render trajectory during simulation")
    parser.add_argument("--render-every", type=int, default=10, help="Render every N steps")
    parser.add_argument("--animate", action="store_true", help="Show final animation after run")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    sim = run(
        num_steps=args.num_steps,
        dt=args.dt,
        initial_state=(args.x0, args.y0, args.th0),
        v_l=args.v_l,
        v_r=args.v_r,
        command_mode=args.command_mode,
        render=args.render,
        render_every=args.render_every,
    )

    final_state = sim.pose
    print(
        "Final state:",
        f"x={final_state[0]:.3f}",
        f"y={final_state[1]:.3f}",
        f"theta={final_state[2]:.3f}",
        f"t={sim.t:.3f}",
    )

    if args.animate:
        Visualizer(sim=sim).render(realtime=True, repeat=True)


if __name__ == "__main__":
    main()
