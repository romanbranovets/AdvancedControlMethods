import numpy as np


def _wrap_angle(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi
