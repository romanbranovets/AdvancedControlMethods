# src/controller.py
"""
Контроллеры для двухмассового активного гасителя вибраций.

MPCController:
    - Линейный MPC с condensing-формулировкой задачи QP.
    - Терминальный вес P берётся из решения дискретного уравнения Риккати
      (бесконечно-горизонтная LQR-стоимость) — гарантирует устойчивость
      замкнутой системы при достаточном горизонте.
    - Все матрицы (H, Sx, Su, SuTQ) пересчитываются один раз в set_model().
    - Решение QP: сначала пробуется аналитический безусловный минимум
      u* = -H^{-1} f, если он лежит внутри ограничений — задача решена за O(N^3)
      без итераций. Если нет — добивается L-BFGS-B с явным якобианом и
      тёплым стартом от предыдущего хода (стандартный приём в реальном MPC).
    - Опциональный feedforward по известному гармоническому возмущению.

LQRController:
    - Бесконечно-горизонтный дискретный LQR. Используется как "честный" эталон
      линейно-оптимального управления без ограничений и без MPC-обвязки.

PIDController:
    - Классический ПИД по положению основной массы.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve_discrete_are


class MPCController:
    def __init__(self, horizon=20, dt=0.05, Q=None, R=None,
                 u_min=-10.0, u_max=10.0,
                 terminal_lqr=True, dist_feedforward=None):
        self.N = horizon
        self.dt = dt
        self.u_min = u_min
        self.u_max = u_max
        if Q is None:
            Q = np.diag([100.0, 1.0, 10.0, 1.0])
        if R is None:
            R = np.array([[1.0]])
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.terminal_lqr = terminal_lqr
        # dist_feedforward: dict {'B_d': ndarray(n,1), 'd': callable(t)->float}
        self.dist_feedforward = dist_feedforward
        self.Ad = None
        self.Bd = None
        self._H = None
        self._H_inv = None
        self._Sx = None
        self._Su = None
        self._SuTQ = None
        self._u_prev = None
        # Для диагностики
        self.last_solve_unconstrained = False
        self.terminal_P = None
        self.lqr_K = None

    def set_model(self, Ad, Bd):
        self.Ad = np.asarray(Ad, dtype=float)
        self.Bd = np.asarray(Bd, dtype=float)
        self._precompute()

    def _precompute(self):
        n = self.Ad.shape[0]
        m = self.Bd.shape[1]
        N = self.N

        # Матрицы предсказания: X = Sx x0 + Su U
        Sx = np.zeros((N * n, n))
        Su = np.zeros((N * n, N * m))
        Apow = np.eye(n)
        for k in range(N):
            Apow = self.Ad @ Apow              # A^{k+1}
            Sx[k*n:(k+1)*n, :] = Apow
        # Su — нижняя блок-теплицева структура
        for j in range(N):
            block = self.Bd.copy()
            Su[j*n:(j+1)*n, j*m:(j+1)*m] = block
            for k in range(j + 1, N):
                block = self.Ad @ block
                Su[k*n:(k+1)*n, j*m:(j+1)*m] = block

        # Терминальный вес P из дискретного Риккати
        if self.terminal_lqr:
            P = solve_discrete_are(self.Ad, self.Bd, self.Q, self.R)
            BtPB_R = self.Bd.T @ P @ self.Bd + self.R
            K = np.linalg.solve(BtPB_R, self.Bd.T @ P @ self.Ad)
            self.terminal_P = P
            self.lqr_K = K
        else:
            P = self.Q.copy()
            self.terminal_P = P
            self.lqr_K = None

        Q_blk = np.zeros((N*n, N*n))
        for i in range(N - 1):
            Q_blk[i*n:(i+1)*n, i*n:(i+1)*n] = self.Q
        Q_blk[(N-1)*n:N*n, (N-1)*n:N*n] = P

        R_blk = np.kron(np.eye(N), self.R)

        H = Su.T @ Q_blk @ Su + R_blk
        H = 0.5 * (H + H.T)  # симметризация против численных артефактов
        # Чуть регуляризуем для устойчивости обратимости
        H += 1e-10 * np.eye(N * m)

        self._Sx = Sx
        self._Su = Su
        self._Q_blk = Q_blk
        self._H = H
        self._H_inv = np.linalg.inv(H)
        self._SuTQ = Su.T @ Q_blk

    def _predict_disturbance_trajectory(self, t0):
        """Прогноз отклика системы на известное возмущение под u=0."""
        if self.dist_feedforward is None:
            return None
        n = self.Ad.shape[0]
        N = self.N
        B_d = np.asarray(self.dist_feedforward['B_d'], dtype=float).reshape(n)
        d_func = self.dist_feedforward['d']
        x_d = np.zeros((N, n))
        x_prev = np.zeros(n)
        for k in range(N):
            d_k = float(d_func(t0 + k * self.dt))
            x_prev = self.Ad @ x_prev + B_d * d_k
            x_d[k] = x_prev
        return x_d.reshape(-1)

    def update(self, state, ref, dt, t=0.0):
        if self.Ad is None or self.Bd is None:
            raise RuntimeError("Model not set. Call set_model first.")
        n = self.Ad.shape[0]
        m = self.Bd.shape[1]
        N = self.N
        x0 = np.asarray(state, dtype=float).reshape(n)
        ref = np.asarray(ref, dtype=float).reshape(n)
        ref_stack = np.tile(ref, N)

        x_dist = self._predict_disturbance_trajectory(t)
        free_resp = self._Sx @ x0
        if x_dist is not None:
            free_resp = free_resp + x_dist

        f = self._SuTQ @ (free_resp - ref_stack)
        # Безусловное решение
        u_unc = -self._H_inv @ f

        if np.all(u_unc >= self.u_min - 1e-9) and np.all(u_unc <= self.u_max + 1e-9):
            u_opt = np.clip(u_unc, self.u_min, self.u_max)
            self.last_solve_unconstrained = True
        else:
            self.last_solve_unconstrained = False
            # Тёплый старт: сдвиг предыдущего решения, либо клиппинг безусловного
            if self._u_prev is not None and len(self._u_prev) == N * m:
                u0 = np.concatenate([self._u_prev[m:], self._u_prev[-m:]])
            else:
                u0 = np.clip(u_unc, self.u_min, self.u_max)
            bounds = [(self.u_min, self.u_max)] * (N * m)
            H = self._H

            def fun(u):
                return 0.5 * u @ H @ u + f @ u

            def jac(u):
                return H @ u + f

            res = minimize(fun, u0, jac=jac, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            u_opt = res.x

        self._u_prev = u_opt.copy()
        return np.array([float(u_opt[0])])

    def reset(self):
        self._u_prev = None
        self.last_solve_unconstrained = False


class LQRController:
    """Бесконечно-горизонтный дискретный LQR (эталон линейно-оптимального управления)."""
    def __init__(self, Q=None, R=None, u_min=-10.0, u_max=10.0):
        if Q is None:
            Q = np.diag([100.0, 1.0, 10.0, 1.0])
        if R is None:
            R = np.array([[1.0]])
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.u_min = u_min
        self.u_max = u_max
        self.K = None
        self.P = None

    def set_model(self, Ad, Bd):
        self.P = solve_discrete_are(Ad, Bd, self.Q, self.R)
        BtPB_R = Bd.T @ self.P @ Bd + self.R
        self.K = np.linalg.solve(BtPB_R, Bd.T @ self.P @ Ad)

    def update(self, state, ref, dt, t=0.0):
        x = np.asarray(state, dtype=float).reshape(-1)
        r = np.asarray(ref, dtype=float).reshape(-1)
        u = -self.K @ (x - r)
        u = np.clip(u, self.u_min, self.u_max)
        return np.array([float(u[0])])

    def reset(self):
        pass


class PIDController:
    """ПИД-регулятор по положению основной массы (x1)."""
    def __init__(self, kp=30.0, ki=8.0, kd=3.0, dt=0.05,
                 u_min=-10.0, u_max=10.0, integral_limit=5.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.u_min = u_min
        self.u_max = u_max
        self.integral_limit = integral_limit
        self._integral = 0.0
        self._prev_error = 0.0

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0

    def update(self, state, ref, dt, t=0.0):
        error = ref[0] - state[0]
        self._integral += error * dt
        self._integral = np.clip(self._integral, -self.integral_limit, self.integral_limit)
        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error
        u = self.kp * error + self.ki * self._integral + self.kd * derivative
        u = np.clip(u, self.u_min, self.u_max)
        return np.array([float(u)])
