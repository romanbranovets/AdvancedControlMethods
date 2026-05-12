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


def _allocate_thrust_pitch(m, g, ax_des, az_des, theta_lim):
    """Желаемые суммарная тяга и тангаж из заданного ускорения (инерциальная СК, z вверх)."""
    fx = m * ax_des
    fz = m * (az_des + g)
    T_des = float(np.hypot(fx, fz))
    theta_des = float(np.arctan2(fx, fz))
    theta_des = float(np.clip(theta_des, -theta_lim, theta_lim))
    return T_des, theta_des


def _inverse_motors(T_des, tau_des, L_arm, k_F, I_min, I_max):
    """
    T = k_F (I_L^2 + I_R^2),  tau = L_arm * k_F * (I_R^2 - I_L^2).
    Возвращает (I_L_des, I_R_des).
    """
    k = k_F
    if k <= 0 or L_arm <= 0:
        return np.array([I_min, I_min], dtype=float)

    a = T_des / k
    b = tau_des / (L_arm * k)
    s_r = 0.5 * (a + b)
    s_l = 0.5 * (a - b)
    eps = 1e-6
    s_r = max(s_r, eps)
    s_l = max(s_l, eps)
    I_R = float(np.clip(np.sqrt(s_r), I_min, I_max))
    I_L = float(np.clip(np.sqrt(s_l), I_min, I_max))
    return np.array([I_L, I_R], dtype=float)


class BacksteppingVelocityController:
    """
    Backstepping по скорости для планарного дрона с двумя моторами (токи).

    Принимает:
        v_des  — целевая скорость (2,) [vx, vz]
        v      — текущая скорость (2,)
        a      — измеренное линейное ускорение (2,) [ax, az] (м/с²)
        tau    — измеренный момент по оси pitch [Н·м]
        I      — измеренные токи (2,) [I_L, I_R] [А]
        theta  — текущий угол тангажа [рад]
        omega  — текущая угловая скорость [рад/с]

    Возвращает:
        I_cmd  — команды токов (2,) [А]
    """

    def __init__(
        self,
        m=0.5,
        g=9.81,
        J=0.015,
        L_arm=0.18,
        k_F=0.35,
        tau_m=0.08,
        kv=2.2,
        k_theta=4.5,
        k_omega=3.5,
        k_I=12.0,
        k_tau_blend=0.15,
        a_lim=6.0,
        theta_lim=0.45,
        I_min=0.0,
        I_max=18.0,
    ):
        self.m = float(m)
        self.g = float(g)
        self.J = float(J)
        self.L_arm = float(L_arm)
        self.k_F = float(k_F)
        self.tau_m = float(tau_m)
        self.kv = float(kv)
        self.k_theta = float(k_theta)
        self.k_omega = float(k_omega)
        self.k_I = float(k_I)
        self.k_tau_blend = float(k_tau_blend)
        self.a_lim = float(a_lim)
        self.theta_lim = float(theta_lim)
        self.I_min = float(I_min)
        self.I_max = float(I_max)

        # внутренние переменные
        self._I_des_prev = np.array([0.0, 0.0], dtype=float)
        self._T_des_prev = 0.0
        self._theta_des_prev = 0.0
        self._omega_des_prev = 0.0
        self.last_lyapunov = None
        self.last_lyapunov_terms = None

    def reset(self, I_hover=None):
        self._theta_hat_prev = 0.0
        if I_hover is None:
            self._I_des_prev = np.array([0.0, 0.0], dtype=float)
        else:
            self._I_des_prev = np.asarray(I_hover, dtype=float).reshape(2).copy()
        self._T_des_prev = self.m * self.g
        self._theta_des_prev = 0.0
        self._omega_des_prev = 0.0
        self.last_lyapunov = None
        self.last_lyapunov_terms = None

        # желаемые значения (для логирования)
        self._theta_des = 0.0
        self._omega_des = 0.0
        self._T_des = 0.0
        self._tau_des = 0.0
        self._I_des = np.array([0.0, 0.0], dtype=float)

    def reset(self, I_hover=None):
        self._I_des_prev = np.zeros(2, dtype=float) if I_hover is None else np.asarray(I_hover, dtype=float).reshape(2).copy()
        self._T_des_prev = self.m * self.g
        self._theta_des_prev = 0.0
        self._omega_des_prev = 0.0
        self._theta_des = 0.0
        self._omega_des = 0.0
        self._T_des = 0.0
        self._tau_des = 0.0
        self._I_des = np.array([0.0, 0.0], dtype=float)
        self.last_lyapunov = None
        self.last_lyapunov_terms = None

    def update(self, v_des, v, a, tau, I, dt, theta, omega):
        """
        v_des, v, a: shape (2,)
        tau: float
        I: shape (2,)
        theta: float
        omega: float
        """
        if dt <= 0.0:
            self.last_lyapunov = None
            self.last_lyapunov_terms = None
            return np.clip(np.asarray(I, dtype=float).reshape(2), self.I_min, self.I_max)

        v_des = np.asarray(v_des, dtype=float).reshape(2)
        v = np.asarray(v, dtype=float).reshape(2)
        a = np.asarray(a, dtype=float).reshape(2)
        I = np.asarray(I, dtype=float).reshape(2)
        tau = float(tau)

        # --- внешний контур по скорости ---
        e_v = v_des - v
        a_des = self.kv * e_v
        an = float(np.linalg.norm(a_des))
        if an > self.a_lim and an > 0:
            a_des = a_des * (self.a_lim / an)

        T_des, theta_des = _allocate_thrust_pitch(self.m, self.g, a_des[0], a_des[1], self.theta_lim)
        T_des = float(np.clip(T_des, 2.0 * self.k_F * self.I_min ** 2 + 1e-6, 2.0 * self.k_F * self.I_max ** 2))

        T_des_dot = (T_des - self._T_des_prev) / dt
        theta_des_dot = (theta_des - self._theta_des_prev) / dt
        self._T_des_prev = T_des
        self._theta_des_prev = theta_des

        # --- контур тангажа (используем переданные theta, omega) ---
        e_th = theta_des - theta
        omega_des = self.k_theta * e_th + theta_des_dot
        omega_des_dot = (omega_des - self._omega_des_prev) / dt
        self._omega_des_prev = omega_des

        tau_ff = self.J * omega_des_dot
        tau_fb = self.J * self.k_omega * (omega_des - omega)
        tau_des = tau_ff + tau_fb
        # мягкая подстройка по измеренному моменту
        tau_des = (1.0 - self.k_tau_blend) * tau_des + self.k_tau_blend * tau

        # --- обратная задача: желаемые токи ---
        I_des = _inverse_motors(T_des, tau_des, self.L_arm, self.k_F, self.I_min, self.I_max)

        I_des_dot = (I_des - self._I_des_prev) / dt
        self._I_des_prev = I_des.copy()

        # --- Lyapunov-like certificate (quadratic in tracking errors) ---
        e_I = I - I_des
        e_om = omega_des - omega
        c_th = self.m * self.g * 0.08
        c_I = self.k_I * self.tau_m
        V_vel = 0.5 * self.m * float(np.dot(e_v, e_v))
        V_pitch = 0.5 * c_th * (e_th ** 2)
        V_omega = 0.5 * self.J * (e_om ** 2)
        V_curr = 0.5 * c_I * float(np.dot(e_I, e_I))
        self.last_lyapunov = float(V_vel + V_pitch + V_omega + V_curr)
        self.last_lyapunov_terms = {
            'V_vel': V_vel,
            'V_pitch': V_pitch,
            'V_omega': V_omega,
            'V_current': V_curr,
        }

        # Backstepping on motor-current dynamics
        I_cmd = I + self.tau_m * (I_des_dot - self.k_I * (I - I_des))
        I_cmd = np.clip(I_cmd, self.I_min, self.I_max)

        # сохраняем желаемые значения для логирования
        self._theta_des = theta_des
        self._omega_des = omega_des
        self._T_des = T_des
        self._tau_des = tau_des
        self._I_des = I_des.copy()

        return I_cmd

    def get_desired(self):
        """Возвращает словарь с желаемыми величинами, вычисленными на последнем шаге."""
        return {
            'theta_des': self._theta_des,
            'omega_des': self._omega_des,
            'T_des': self._T_des,
            'tau_des': self._tau_des,
            'I_des': self._I_des.copy()
        }


class _VelocityMotorCascadeBase:
    """
    Shared cascade: desired translational acceleration -> (T_des, theta_des)
    -> inner torque PD on estimated pitch -> motor inversion.
    Subclasses implement _accel_des(v_des, v, a, dt).
    """

    def __init__(self, m, g, J, L_arm, k_F, k_th, k_w, a_lim, theta_lim, I_min, I_max):
        self.m = float(m)
        self.g = float(g)
        self.J = float(J)
        self.L_arm = float(L_arm)
        self.k_F = float(k_F)
        self.k_th = float(k_th)
        self.k_w = float(k_w)
        self.a_lim = float(a_lim)
        self.theta_lim = float(theta_lim)
        self.I_min = float(I_min)
        self.I_max = float(I_max)

        # желаемые величины для логирования
        self._theta_des = 0.0
        self._tau_des = 0.0
        self._T_des = 0.0
        self._I_des = np.array([0.0, 0.0], dtype=float)

    def reset(self, I_hover=None):
        self._theta_hat_prev = 0.0
        self._reset_extra()

    def _reset_extra(self):
        pass

    def _accel_des(self, v_des, v, a, dt):
        raise NotImplementedError

    def update(self, v_des, v, a, tau, I, dt, theta, omega):
        """
        v_des, v, a: shape (2,)
        tau, theta, omega: float
        I: shape (2,)
        """
        v_des = np.asarray(v_des, dtype=float).reshape(2)
        v = np.asarray(v, dtype=float).reshape(2)
        a = np.asarray(a, dtype=float).reshape(2)
        if dt <= 0.0:
            return np.clip(np.asarray(I, dtype=float).reshape(2), self.I_min, self.I_max)

        az_p = float(a[1] + self.g)
        theta_hat = float(np.arctan2(a[0], np.sign(az_p) * max(abs(az_p), 1e-3)))
        omega_hat = (theta_hat - self._theta_hat_prev) / dt
        self._theta_hat_prev = theta_hat

        a_des = self._accel_des(v_des, v, a, dt)
        an = float(np.linalg.norm(a_des))
        if an > self.a_lim and an > 0:
            a_des = a_des * (self.a_lim / an)

        T_des, theta_des = _allocate_thrust_pitch(self.m, self.g, a_des[0], a_des[1], self.theta_lim)
        T_des = float(np.clip(T_des, 1e-3, 2.0 * self.k_F * self.I_max ** 2))

        e_th = theta_des - theta_hat
        tau_des = self.J * (self.k_th * e_th - self.k_w * omega_hat)
        tau_max = self.L_arm * self.k_F * self.I_max ** 2
        tau_des = float(np.clip(tau_des, -tau_max, tau_max))

        I_cmd = _inverse_motors(T_des, tau_des, self.L_arm, self.k_F, self.I_min, self.I_max)
        return np.clip(I_cmd, self.I_min, self.I_max)


class PVelocityMotorController(_VelocityMotorCascadeBase):
    """Outer loop: proportional on velocity error only (constant v_des assumed for tuning)."""

    def __init__(self, m, g, J, L_arm, k_F, kp, k_th, k_w, a_lim, theta_lim, I_min, I_max):
        super().__init__(m, g, J, L_arm, k_F, k_th, k_w, a_lim, theta_lim, I_min, I_max)
        self.kp = float(kp)

    def _accel_des(self, v_des, v, a, dt):
        return self.kp * (v_des - v)


class PDVelocityMotorController(_VelocityMotorCascadeBase):
    """
    Outer loop: P on velocity + D using inertial acceleration (for constant v_des, de/dt = -a).
    """

    def __init__(self, m, g, J, L_arm, k_F, kp, kd, k_th, k_w, a_lim, theta_lim, I_min, I_max):
        super().__init__(m, g, J, L_arm, k_F, k_th, k_w, a_lim, theta_lim, I_min, I_max)
        self.kp = float(kp)
        self.kd = float(kd)

    def _accel_des(self, v_des, v, a, dt):
        e = v_des - v
        return self.kp * e - self.kd * a


class PIDVelocityMotorController(_VelocityMotorCascadeBase):
    """Outer loop: PD as above + integral on velocity error (with clamp)."""

    def __init__(self, m, g, J, L_arm, k_F, kp, kd, ki, k_th, k_w, a_lim, theta_lim,
                 I_min, I_max, integral_limit=2.5):
        super().__init__(m, g, J, L_arm, k_F, k_th, k_w, a_lim, theta_lim, I_min, I_max)
        self.kp = float(kp)
        self.kd = float(kd)
        self.ki = float(ki)
        self.integral_limit = float(integral_limit)
        self._int_e = np.zeros(2, dtype=float)

    def _reset_extra(self):
        self._int_e[:] = 0.0

    def _accel_des(self, v_des, v, a, dt):
        e = v_des - v
        self._int_e += e * dt
        self._int_e = np.clip(self._int_e, -self.integral_limit, self.integral_limit)
        return self.kp * e - self.kd * a + self.ki * self._int_e
