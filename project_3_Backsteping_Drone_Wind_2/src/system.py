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


class PlanarDrone2D:
    """
    Планарный дрон в плоскости (x, z): два мотора по оси x_body, тяга в +z_body.

    Состояние (8):
        x, z — положение ЦМ [м]
        vx, vz — скорость [м/с]
        theta — угол тангажа [рад] (нос вверх положителен), ось вращения +y
        omega — угловая скорость [рад/с]
        I_L, I_R — токи левого/правого мотора [А]

    Управление: u = [I_L_cmd, I_R_cmd] — команды тока [А].
    Тяга каждого мотора: T_i = k_F * I_i^2 (I_i >= 0).
    Суммарная тяга вдоль +z_body: T = T_L + T_R.
    Момент от разницы тяг (плечо L): tau = (L/2) * (T_R - T_L) * 2 / ... 
    Фактически tau_y = L * k_F * (I_R^2 - I_L^2).

    Уравнения (инерциальная СК, z вверх):
        m * dvx/dt = T*sin(theta) + w_x - c_d * vx
        m * dvz/dt = T*cos(theta) - m*g + w_z - c_d * vz
        J * domega/dt = tau + w_tau
        dtheta/dt = omega
        dI/dt = (1/tau_m) * (I_cmd - I)  (I >= 0, насыщение на нуле)
    """

    def __init__(
        self,
        m=0.5,
        g=9.81,
        J=0.015,
        L_arm=0.18,
        k_F=0.35,
        tau_m=0.08,
        I_min=0.0,
        I_max=18.0,
        c_d=0.05,
    ):
        self.m = float(m)
        self.g = float(g)
        self.J = float(J)
        self.L_arm = float(L_arm)
        self.k_F = float(k_F)
        self.tau_m = float(tau_m)
        self.I_min = float(I_min)
        self.I_max = float(I_max)
        self.c_d = float(c_d)

    def thrust_torque(self, I_L, I_R):
        T_L = self.k_F * I_L * I_L
        T_R = self.k_F * I_R * I_R
        T = T_L + T_R
        tau = self.L_arm * self.k_F * (I_R * I_R - I_L * I_L)
        return T, tau

    def dynamics(self, state, t, u, wind_func=None):
        """
        state: (8,)  [x, z, vx, vz, theta, omega, I_L, I_R]
        u: (2,) команды тока
        wind_func(t) -> (wx, wz, w_tau) или (wx, wz) — силы [Н] и опционально момент [Н·м]
        """
        u = np.asarray(u, dtype=float).reshape(2)
        u_L = float(np.clip(u[0], self.I_min, self.I_max))
        u_R = float(np.clip(u[1], self.I_min, self.I_max))

        x, z, vx, vz, theta, omega, I_L, I_R = state
        I_L = max(float(I_L), 0.0)
        I_R = max(float(I_R), 0.0)

        if wind_func is not None:
            w = wind_func(t)
            w = np.asarray(w, dtype=float).reshape(-1)
            wx, wz = float(w[0]), float(w[1])
            w_tau = float(w[2]) if w.size >= 3 else 0.0
        else:
            wx = wz = w_tau = 0.0

        T, tau = self.thrust_torque(I_L, I_R)

        st, ct = np.sin(theta), np.cos(theta)
        ax = (T * st + wx) / self.m - (self.c_d / self.m) * vx
        az = (T * ct - self.m * self.g + wz) / self.m - (self.c_d / self.m) * vz
        alpha = tau / self.J + w_tau / self.J

        dI_L = self._d_current(I_L, u_L)
        dI_R = self._d_current(I_R, u_R)

        return np.array([vx, vz, ax, az, omega, alpha, dI_L, dI_R], dtype=float)

    def _d_current(self, I, u_cmd):
        if I <= self.I_min and (u_cmd - I) < 0.0:
            return 0.0
        return (1.0 / self.tau_m) * (u_cmd - I)

    @staticmethod
    def hover_currents(m, g, k_F):
        """Симметричные токи удержания высоты (theta=0)."""
        T_need = m * g
        s = max(T_need / (2.0 * k_F), 0.0)
        Ih = np.sqrt(s)
        return np.array([Ih, Ih], dtype=float)
