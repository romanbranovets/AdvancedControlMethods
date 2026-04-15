import numpy as np


def wrap_to_pi(a: float) -> float:
    """Wrap an angle in radians to the interval [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def _wrap_angle(a: float) -> float:
    return wrap_to_pi(a)
