# src/scenarios.py
"""Сценарии возмущения для честных экспериментов."""
import numpy as np


def no_disturbance():
    return lambda t: 0.0


def sine(amplitude, freq):
    omega = 2 * np.pi * freq
    return lambda t: amplitude * np.sin(omega * t)


def chirp(amplitude, f0, f1, t1):
    """Линейный chirp от f0 Гц до f1 Гц за t1 секунд."""
    def fn(t):
        # Мгновенная фаза для линейного chirp: phi(t) = 2π(f0*t + 0.5*k*t^2), k=(f1-f0)/t1
        k = (f1 - f0) / t1
        phi = 2 * np.pi * (f0 * t + 0.5 * k * t * t)
        return amplitude * np.sin(phi)
    return fn


def step(amplitude, t_on=1.0):
    return lambda t: amplitude if t >= t_on else 0.0
