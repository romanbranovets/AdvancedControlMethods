# src/controller.py
import numpy as np
from scipy.optimize import minimize

class MPCController:
    """
    MPC для двухмассовой системы (состояние 4, управление 1).
    """
    def __init__(self, horizon=20, dt=0.05, Q=None, R=None,
                 u_min=-10.0, u_max=10.0):
        self.N = horizon
        self.dt = dt
        self.u_min = u_min
        self.u_max = u_max
        if Q is None:
            Q = np.diag([100.0, 1.0, 10.0, 1.0])
        if R is None:
            R = np.array([[1.0]])
        self.Q = Q
        self.R = R
        self.Ad = None
        self.Bd = None

    def set_model(self, Ad, Bd):
        self.Ad = Ad
        self.Bd = Bd

    def _build_qp(self, x0, ref):
        n = self.Ad.shape[0]      # 4
        m = self.Bd.shape[1]      # 1

        Sx = np.zeros((self.N * n, n))
        Su = np.zeros((self.N * n, self.N * m))

        for k in range(self.N):
            Ak = np.linalg.matrix_power(self.Ad, k)
            Sx[k*n:(k+1)*n, :] = Ak
            for j in range(k):
                Aj = np.linalg.matrix_power(self.Ad, k-1-j)
                Su[k*n:(k+1)*n, j*m:(j+1)*m] = Aj @ self.Bd

        Q_blk = np.kron(np.eye(self.N), self.Q)
        R_blk = np.kron(np.eye(self.N), self.R)
        ref_stack = np.tile(ref, self.N)

        H = Su.T @ Q_blk @ Su + R_blk
        f = (Sx @ x0 - ref_stack).T @ Q_blk @ Su

        lb = np.full(self.N * m, self.u_min)
        ub = np.full(self.N * m, self.u_max)

        return H, f, lb, ub

    def update(self, state, ref, dt):
        if self.Ad is None or self.Bd is None:
            raise RuntimeError("Model not set. Call set_model first.")
        x0 = np.asarray(state, dtype=float).reshape(4)
        ref = np.asarray(ref, dtype=float).reshape(4)

        H, f, lb, ub = self._build_qp(x0, ref)
        n_u = self.N * 1
        u0 = np.zeros(n_u)
        bounds = [(lb[i], ub[i]) for i in range(n_u)]

        def objective(u):
            return 0.5 * u @ H @ u + f @ u

        res = minimize(objective, u0, method='SLSQP', bounds=bounds,
                       options={'maxiter': 200, 'ftol': 1e-8})
        if res.success:
            u_opt = res.x[0]
        else:
            u_opt = 0.0
        return np.array([u_opt])

    def reset(self):
        pass


class PIDController:
    """ПИД‑регулятор, управляющий только по положению основной массы."""
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

    def update(self, state, ref, dt):
        error = ref[0] - state[0]        # только по x1
        self._integral += error * dt
        self._integral = np.clip(self._integral, -self.integral_limit, self.integral_limit)
        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error
        u = self.kp * error + self.ki * self._integral + self.kd * derivative
        u = np.clip(u, self.u_min, self.u_max)
        return np.array([u])