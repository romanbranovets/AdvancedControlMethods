# src/system.py
import numpy as np

class ActiveVibrationAbsorber2D:
    """
    Двухмассовая система активного динамического гасителя вибраций.
    Состояние: [x1, x2, v1, v2]
        x1 – положение основной массы m1
        x2 – положение гасителя m2
    Управление: u – сила актуатора (приложена между массами)
    Возмущение: гармоническая сила, действующая на основную массу.

    Уравнения движения:
        m1 * x1'' + c1 * x1' + k1 * x1 + c2 * (x1' - x2') + k2 * (x1 - x2) = u + d(t)
        m2 * x2'' + c2 * (x2' - x1') + k2 * (x2 - x1) = -u
    """

    def __init__(self, m1=1.0, m2=0.2, k1=10.0, k2=5.0,
                 c1=0.5, c2=0.1, dist_amplitude=2.0, dist_freq=3.0,
                 u_min=-10.0, u_max=10.0):
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

    def disturbance(self, t):
        """Внешняя гармоническая сила на основной массе."""
        return self.dist_amplitude * np.sin(2 * np.pi * self.dist_freq * t)

    def dynamics(self, state, t, u):
        """Правая часть ОДУ."""
        x1, x2, v1, v2 = state
        u = np.clip(u, self.u_min, self.u_max).item()
        d = self.disturbance(t)

        # Силы пружин и демпферов
        F_spring1 = -self.k1 * x1
        F_damp1   = -self.c1 * v1
        F_spring2 = -self.k2 * (x1 - x2)
        F_damp2   = -self.c2 * (v1 - v2)

        # Ускорения
        a1 = (F_spring1 + F_damp1 + F_spring2 + F_damp2 + u + d) / self.m1
        a2 = (-F_spring2 - F_damp2 - u) / self.m2

        return np.array([v1, v2, a1, a2])

    def get_linear_matrices(self):
        """
        Возвращает матрицы A (4x4) и B (4x1) линеаризованной системы.
        Состояние: [x1, x2, v1, v2].
        """
        A = np.array([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-(self.k1 + self.k2)/self.m1, self.k2/self.m1,
             -(self.c1 + self.c2)/self.m1, self.c2/self.m1],
            [self.k2/self.m2, -self.k2/self.m2,
             self.c2/self.m2, -self.c2/self.m2]
        ])
        B = np.array([
            [0.0],
            [0.0],
            [1.0/self.m1],
            [-1.0/self.m2]
        ])
        return A, B

    @staticmethod
    def linear_discrete(A, B, dt):
        """Дискретизация методом разложения в ряд Тейлора 3‑го порядка."""
        n = A.shape[0]
        m_cols = B.shape[1]
        M = np.block([[A, B],
                      [np.zeros((m_cols, n)), np.zeros((m_cols, m_cols))]])
        Md = np.eye(n + m_cols) + M * dt + M @ M * dt**2 / 2.0 + M @ M @ M * dt**3 / 6.0
        Ad = Md[:n, :n]
        Bd = Md[:n, n:]
        return Ad, Bd