# src/system.py
"""
Двухмассовая модель активного динамического гасителя вибраций.

Состояние: x = [x1, x2, v1, v2]^T
    x1, v1 — положение и скорость основной массы m1
    x2, v2 — положение и скорость гасителя m2

Уравнения движения (физически согласованные знаки):
    m1 v1' = -k1*x1 - c1*v1 + k2*(x2 - x1) + c2*(v2 - v1) + u + d(t)
    m2 v2' = -k2*(x2 - x1) - c2*(v2 - v1) - u

В таком виде линеаризация совпадает с моделью, используемой MPC.
"""
import numpy as np
from scipy.linalg import expm


class ActiveVibrationAbsorber2D:
    def __init__(self, m1=1.0, m2=0.2, k1=10.0, k2=5.0,
                 c1=0.5, c2=0.1,
                 dist_amplitude=2.0, dist_freq=3.0,
                 u_min=-10.0, u_max=10.0,
                 disturbance_fn=None):
        self.m1 = m1
        self.m2 = m2
        self.k1 = k1
        self.k2 = k2
        self.c1 = c1
        self.c2 = c2
        self.dist_amplitude = dist_amplitude
        self.dist_freq = dist_freq
        self.u_min = u_min
        self.u_max = u_max
        # Возможность подменить возмущение (sin / chirp / step / off)
        self._disturbance_fn = disturbance_fn

    # ---- возмущение ----------------------------------------------------------
    def disturbance(self, t):
        if self._disturbance_fn is not None:
            return float(self._disturbance_fn(t))
        return self.dist_amplitude * np.sin(2 * np.pi * self.dist_freq * t)

    def set_disturbance(self, fn):
        self._disturbance_fn = fn

    # ---- нелинейная (здесь — линейная по сути) динамика ---------------------
    def dynamics(self, state, t, u):
        x1, x2, v1, v2 = state
        u = float(np.clip(u, self.u_min, self.u_max))
        d = self.disturbance(t)
        a1 = (-self.k1 * x1 - self.c1 * v1
              + self.k2 * (x2 - x1) + self.c2 * (v2 - v1)
              + u + d) / self.m1
        a2 = (-self.k2 * (x2 - x1) - self.c2 * (v2 - v1) - u) / self.m2
        return np.array([v1, v2, a1, a2])

    # ---- линейные матрицы непрерывной системы --------------------------------
    def get_linear_matrices(self):
        """Непрерывные матрицы A (4x4), B (4x1) и B_d (4x1) для возмущения."""
        A = np.array([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-(self.k1 + self.k2) / self.m1, self.k2 / self.m1,
             -(self.c1 + self.c2) / self.m1, self.c2 / self.m1],
            [self.k2 / self.m2, -self.k2 / self.m2,
             self.c2 / self.m2, -self.c2 / self.m2]
        ])
        B = np.array([[0.0], [0.0], [1.0 / self.m1], [-1.0 / self.m2]])
        B_d = np.array([[0.0], [0.0], [1.0 / self.m1], [0.0]])
        return A, B, B_d

    @staticmethod
    def linear_discrete(A, B, dt):
        """
        Точная дискретизация ZOH через матричную экспоненту:
            [Ad Bd] = expm( [[A B],[0 0]] * dt )[:n,:]
        """
        n = A.shape[0]
        m = B.shape[1]
        M = np.zeros((n + m, n + m))
        M[:n, :n] = A
        M[:n, n:] = B
        Md = expm(M * dt)
        Ad = Md[:n, :n]
        Bd = Md[:n, n:]
        return Ad, Bd
