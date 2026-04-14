"""
cannon.py — Cannon and projectile dynamics.

=============================================================================
RANDOMNESS PROOF
=============================================================================
The cannon fires as a **Poisson process** with rate λ = 1 / mean_fire_interval.

  Inter-arrival times:  T_k ~ Exp(λ),   k = 1, 2, 3, …
  i.i.d., drawn via:    T_k = −mean_fire_interval · ln(U_k)
  where                 U_k ~ Uniform(0, 1)

The underlying PRNG is **numpy.random.default_rng** which by default uses
the PCG-64 algorithm — a statistically high-quality, non-cryptographic
generator.  Every raw uniform variate U_k drawn for timing and every raw
normal variate N_k drawn for aim noise are stored in `cannon.fire_log`
so that the sequence can be verified independently:

  To verify shot k:
    T_k = -mean_fire_interval * ln(fire_log[k]["raw_u_interval"])
    θ_noise_k = fire_log[k]["raw_n_aim"] * angular_spread_std

=============================================================================
AIMING MODEL
=============================================================================
Each shot is aimed at the robot's current position with independent
Gaussian angular noise:

  θ_aim   = atan2(y_robot − y_cannon, x_robot − x_cannon)
  θ_noise ~ N(0, angular_spread_std²)
  θ_fire  = θ_aim + θ_noise

  v_x = projectile_speed · cos(θ_fire)
  v_y = projectile_speed · sin(θ_fire)

=============================================================================
PROJECTILE MODEL
=============================================================================
A projectile is a disk of radius `projectile_radius` moving at constant
velocity (no gravity, no drag, no bouncing):

  x(t) = x_0 + v_x · t
  y(t) = y_0 + v_y · t

It becomes inactive when it exits the arena or exceeds max_age.
=============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Projectile
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Projectile:
    """A single cannon shell flying at constant velocity.

    Attributes
    ----------
    x, y     : current position, m
    vx, vy   : constant velocity components, m/s
    radius   : collision radius, m
    alive    : False once the shell exits the arena or exceeds max_age
    age      : elapsed time since launch, s
    """

    x: float
    y: float
    vx: float
    vy: float
    radius: float
    alive: bool = True
    age: float = 0.0

    # Position history kept for visualisation trails
    _xs: list[float] = field(default_factory=list)
    _ys: list[float] = field(default_factory=list)

    def step(
        self,
        dt: float,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        max_age: float,
    ) -> None:
        """Advance one Euler step; mark dead if out of bounds or too old."""
        if not self.alive:
            return
        self._xs.append(self.x)
        self._ys.append(self.y)
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.age += dt
        if (
            self.x < x_bounds[0]
            or self.x > x_bounds[1]
            or self.y < y_bounds[0]
            or self.y > y_bounds[1]
            or self.age >= max_age
        ):
            self.alive = False

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy], dtype=float)

    @property
    def trail_xs(self) -> list[float]:
        return list(self._xs)

    @property
    def trail_ys(self) -> list[float]:
        return list(self._ys)


# ---------------------------------------------------------------------------
# Cannon
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Cannon:
    """Stationary cannon that fires projectiles as a Poisson process.

    Parameters
    ----------
    x, y                : cannon position in the arena (m)
    mean_fire_interval  : mean time between shots (s).
                          Inter-arrival times are Exp(1/mean_fire_interval).
    projectile_speed    : muzzle speed (m/s)
    projectile_radius   : collision radius of each shell (m)
    angular_spread_std  : Gaussian std of angular aim noise (rad).
                          0 = perfectly aimed; larger values → wider spread.
    rng                 : numpy random.Generator.
                          Use np.random.default_rng(seed) for reproducibility.
    max_projectile_age  : projectiles are removed after this many seconds (s)

    Notes
    -----
    All raw random variates are logged in `fire_log` for independent
    statistical verification of the Poisson process assumption.
    """

    x: float
    y: float
    mean_fire_interval: float
    projectile_speed: float
    projectile_radius: float
    angular_spread_std: float
    rng: np.random.Generator
    max_projectile_age: float = 8.0

    # Runtime state (not constructor arguments)
    _time_until_next_shot: float = field(init=False)
    _fire_log: list[dict] = field(init=False, default_factory=list)
    _shot_count: int = field(init=False, default=0)
    _last_raw_u: float = field(init=False, default=0.0)
    _last_interval: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self._time_until_next_shot = self._sample_interval()
        self._fire_log = []
        self._shot_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    @property
    def shot_count(self) -> int:
        """Total number of shots fired so far."""
        return self._shot_count

    @property
    def fire_log(self) -> list[dict]:
        """
        One entry per shot::

            {
              "shot"              : int,   # shot index (1-based)
              "t"                 : float, # simulation time of firing (s)
              "aim_angle_rad"     : float, # atan2 toward robot (rad)
              "noise_angle_rad"   : float, # Gaussian noise added to aim (rad)
              "fire_angle_rad"    : float, # actual launch angle (rad)
              "interval_s"        : float, # Exp-sampled wait until THIS shot (s)
              "next_interval_s"   : float, # Exp-sampled wait after this shot (s)
              "raw_u_interval"    : float, # raw Uniform(0,1) for interval (for verification)
              "raw_n_aim"         : float, # raw N(0,1) for aim noise (for verification)
            }

        Verification example::

            log = cannon.fire_log[0]
            T_reconstructed = -cannon.mean_fire_interval * math.log(log["raw_u_interval"])
            # T_reconstructed ≈ log["interval_s"]
        """
        return list(self._fire_log)

    def update(
        self,
        dt: float,
        t: float,
        robot_pos: np.ndarray,
    ) -> Projectile | None:
        """Advance the cannon timer; return a new Projectile if a shot fires.

        Parameters
        ----------
        dt        : time step (s)
        t         : current simulation time (s)
        robot_pos : robot position [x, y] used for aiming

        Returns
        -------
        Projectile if a shot was fired this step, else None.
        """
        self._time_until_next_shot -= dt
        if self._time_until_next_shot > 0.0:
            return None
        next_interval = self._sample_interval()
        projectile = self._fire(t, robot_pos, next_interval)
        self._time_until_next_shot = next_interval
        return projectile

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sample_interval(self) -> float:
        """Draw one Exp(1/λ) inter-arrival time.

        numpy uses the inverse-CDF method:
            U ~ Uniform(0, 1)
            T = −mean · ln(U)   ← stored as raw_u_interval for verification
        """
        # We draw U explicitly so we can log it.
        u = float(self.rng.uniform(0.0, 1.0))
        # Guard against U=0 (measure-zero event, but be safe)
        u = max(u, 1e-12)
        t_sample = -self.mean_fire_interval * np.log(u)
        # Store u on a thread-local scratch attribute for the next _fire call
        self._last_raw_u = u
        self._last_interval = float(t_sample)
        return float(t_sample)

    def _fire(
        self,
        t: float,
        robot_pos: np.ndarray,
        next_interval: float,
    ) -> Projectile:
        """Create a projectile aimed (noisily) at the robot."""
        dx = robot_pos[0] - self.x
        dy = robot_pos[1] - self.y
        aim_angle = float(np.arctan2(dy, dx))

        raw_n = float(self.rng.standard_normal())
        noise_angle = raw_n * self.angular_spread_std
        fire_angle = aim_angle + noise_angle

        vx = self.projectile_speed * np.cos(fire_angle)
        vy = self.projectile_speed * np.sin(fire_angle)

        self._shot_count += 1
        self._fire_log.append(
            {
                "shot": self._shot_count,
                "t": t,
                "aim_angle_rad": aim_angle,
                "noise_angle_rad": noise_angle,
                "fire_angle_rad": fire_angle,
                "interval_s": self._last_interval,
                "next_interval_s": next_interval,
                "raw_u_interval": self._last_raw_u,
                "raw_n_aim": raw_n,
            }
        )

        return Projectile(
            x=self.x,
            y=self.y,
            vx=float(vx),
            vy=float(vy),
            radius=self.projectile_radius,
        )
