# src/controller.py
import numpy as np


class PIDController:
    """Простой ПИД‑регулятор, игнорирующий динамику тяги."""

    def __init__(self, kp=5.0, ki=0.5, kd=3.0, integral_limit=None, u_max=20.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.u_max = u_max
        self._integral = 0.0
        self._prev_error = None

    def reset(self):
        self._integral = 0.0
        self._prev_error = None

    def update(self, state, z_des, dt):
        """Возвращает команду u."""
        z = state[0]
        v = state[1]
        error = z_des - z

        if dt <= 0.0:
            return 0.0

        self._integral += error * dt
        if self.integral_limit is not None:
            self._integral = np.clip(self._integral, -self.integral_limit, self.integral_limit)

        if self._prev_error is None:
            d_error = 0.0
        else:
            d_error = (error - self._prev_error) / dt
        self._prev_error = error

        u = self.kp * error + self.ki * self._integral + self.kd * d_error
        u = np.clip(u, 0.0, self.u_max)
        return u


class BacksteppingController:
    """
    Backstepping‑регулятор для высоты с учётом динамики тяги.
    Эталонное ускорение формируется PD‑законом:
        a_des = kp*(z_des - z) - kd*v
    Желаемая тяга:
        T_des = m * (g + a_des)   (ограничена)
    Управление (backstepping):
        u = T + tau_m * ( T_des_dot - k_T * (T - T_des) )
    """

    def __init__(self, m=0.5, g=9.81, tau_m=0.1,
                 kp=5.0, kd=3.0,
                 k_T=10.0,
                 thrust_min=0.0, thrust_max=20.0,
                 u_max=20.0):
        self.m = m
        self.g = g
        self.tau_m = tau_m
        self.kp = kp
        self.kd = kd
        self.k_T = k_T
        self.thrust_min = thrust_min
        self.thrust_max = thrust_max
        self.u_max = u_max
        self._T_des_prev = m * g   # начальное приближение

    def reset(self):
        self._T_des_prev = self.m * self.g

    def update(self, state, z_des, dt):
        """Возвращает команду u."""
        if dt <= 0.0:
            return 0.0

        z, v, T = state

        # Эталонное ускорение
        a_des = self.kp * (z_des - z) - self.kd * v

        # Желаемая тяга
        T_des = self.m * (self.g + a_des)
        T_des = np.clip(T_des, self.thrust_min, self.thrust_max)

        # Производная желаемой тяги (конечная разность)
        T_des_dot = (T_des - self._T_des_prev) / dt

        # Закон управления
        u_unsat = T + self.tau_m * (T_des_dot - self.k_T * (T - T_des))
        u = np.clip(u_unsat, 0.0, self.u_max)

        self._T_des_prev = T_des
        return u