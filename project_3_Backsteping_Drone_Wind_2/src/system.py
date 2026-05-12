# src/system.py
import numpy as np


class VerticalDrone:
    """
    Одномерная динамика высоты с учётом инерционности тяги.

    Состояние:          state = [z, v, T]
    Управление:         u (скаляр) – команда на мотор
    Внешнее возмущение: w(t) – сила [Н], действующая на дрон (например, ветер)

    Уравнения:
        dz/dt = v
        dv/dt = (T - m*g + w) / m
        dT/dt = (1 / tau_m) * (u - T)   (обрезание снизу нулём)
    """

    def __init__(self, m=0.5, g=9.81, tau_m=0.1, u_min=0.0, u_max=20.0):
        self.m = m
        self.g = g
        self.tau_m = tau_m
        self.u_min = u_min
        self.u_max = u_max

    def dynamics(self, state, t, u, wind_func=None):
        """Возвращает производную состояния [dz, dv, dT]."""
        u = np.clip(float(u), self.u_min, self.u_max)
        z, v, T = state

        # Внешняя сила (ветер)
        if wind_func is not None:
            w = wind_func(t)
        else:
            w = 0.0

        dz = v
        dv = (T - self.m * self.g + w) / self.m

        # Динамика тяги – не допускаем отрицательной тяги
        if T <= 0.0 and (u - T) < 0.0:
            dT = 0.0
        else:
            dT = (1.0 / self.tau_m) * (u - T)

        return np.array([dz, dv, dT])