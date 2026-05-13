# src/controller.py
import numpy as np


class PIDController:
    """Простой ПИД‑регулятор для 1D (высота)."""

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
        self._T_des_prev = m * g

    def reset(self):
        self._T_des_prev = self.m * self.g

    def update(self, state, z_des, dt):
        if dt <= 0.0:
            return 0.0

        z, v, T = state

        a_des = self.kp * (z_des - z) - self.kd * v
        T_des = self.m * (self.g + a_des)
        T_des = np.clip(T_des, self.thrust_min, self.thrust_max)

        T_des_dot = (T_des - self._T_des_prev) / dt

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
    Backstepping контроллер для планарного дрона.
    Принимает целевую позицию, текущее состояние (x,z,vx,vz,theta,omega,I_L,I_R),
    измеренные ускорение и момент.
    Внутри реализован позиционный контур (PD по позиции -> желаемая скорость,
    P по скорости -> желаемое ускорение).
    """

    def __init__(
        self,
        m=0.5,
        g=9.81,
        J=0.015,
        L_arm=0.18,
        k_F=0.35,
        tau_m=0.08,
        # Параметры позиционного контура
        pos_kp=0.8,
        pos_kd=0.4,
        vel_kp=2.4,
        v_max=1.5,
        # Параметры контура тангажа
        k_theta=5.0,
        k_omega=4.0,
        k_I=14.0,
        k_tau_blend=0.12,
        # Ограничения
        a_lim=5.5,
        theta_lim=0.42,
        I_min=0.0,
        I_max=18.0,
    ):
        self.m = float(m)
        self.g = float(g)
        self.J = float(J)
        self.L_arm = float(L_arm)
        self.k_F = float(k_F)
        self.tau_m = float(tau_m)

        self.pos_kp = float(pos_kp)
        self.pos_kd = float(pos_kd)
        self.vel_kp = float(vel_kp)
        self.v_max = float(v_max)

        self.k_theta = float(k_theta)
        self.k_omega = float(k_omega)
        self.k_I = float(k_I)
        self.k_tau_blend = float(k_tau_blend)

        self.a_lim = float(a_lim)
        self.theta_lim = float(theta_lim)
        self.I_min = float(I_min)
        self.I_max = float(I_max)

        # внутренние переменные для желаемых величин
        self._T_des_prev = 0.0
        self._theta_des_prev = 0.0
        self._omega_des_prev = 0.0
        self._I_des_prev = np.array([0.0, 0.0], dtype=float)

        # желаемые величины для логирования
        self._v_des = np.zeros(2, dtype=float)
        self._a_des = np.zeros(2, dtype=float)
        self._theta_des = 0.0
        self._omega_des = 0.0
        self._T_des = 0.0
        self._tau_des = 0.0
        self._I_des = np.zeros(2, dtype=float)

        self.last_lyapunov = None
        self.last_lyapunov_terms = None

    def reset(self, I_hover=None):
        self._T_des_prev = self.m * self.g
        self._theta_des_prev = 0.0
        self._omega_des_prev = 0.0
        self._I_des_prev = np.zeros(2, dtype=float) if I_hover is None else np.asarray(I_hover, dtype=float).copy()
        self._v_des[:] = 0.0
        self._a_des[:] = 0.0
        self._theta_des = 0.0
        self._omega_des = 0.0
        self._T_des = 0.0
        self._tau_des = 0.0
        self._I_des[:] = 0.0
        self.last_lyapunov = None
        self.last_lyapunov_terms = None

    def update(self, state, target_pos, a, tau, dt):
        """
        state: (8,) [x, z, vx, vz, theta, omega, I_L, I_R]
        target_pos: (2,) [x_des, z_des]
        a: (2,) измеренное линейное ускорение [ax, az] (м/с²)
        tau: float измеренный момент тангажа [Н·м]
        dt: float шаг по времени
        """
        if dt <= 0.0:
            self.last_lyapunov = None
            self.last_lyapunov_terms = None
            return np.clip(state[6:8], self.I_min, self.I_max)

        x, z, vx, vz, theta, omega, I_L, I_R = state
        target_x, target_z = target_pos

        # ----- Позиционный контур (PD) -----
        e_pos = np.array([target_x - x, target_z - z], dtype=float)
        v_des = self.pos_kp * e_pos - self.pos_kd * np.array([vx, vz])
        v_norm = np.linalg.norm(v_des)
        if v_norm > self.v_max and v_norm > 0:
            v_des = v_des * (self.v_max / v_norm)

        # ----- Контур скорости (P) -----
        e_v = v_des - np.array([vx, vz])
        a_des = self.vel_kp * e_v
        an = float(np.linalg.norm(a_des))
        if an > self.a_lim and an > 0:
            a_des = a_des * (self.a_lim / an)

        # Желаемая тяга и тангаж
        T_des, theta_des = _allocate_thrust_pitch(self.m, self.g, a_des[0], a_des[1], self.theta_lim)
        T_des = float(np.clip(T_des, 2.0 * self.k_F * self.I_min ** 2 + 1e-6, 2.0 * self.k_F * self.I_max ** 2))

        # Производные желаемых величин
        T_des_dot = (T_des - self._T_des_prev) / dt
        theta_des_dot = (theta_des - self._theta_des_prev) / dt
        self._T_des_prev = T_des
        self._theta_des_prev = theta_des

        # Контур тангажа
        e_th = theta_des - theta
        omega_des = self.k_theta * e_th + theta_des_dot
        omega_des_dot = (omega_des - self._omega_des_prev) / dt
        self._omega_des_prev = omega_des

        tau_ff = self.J * omega_des_dot
        tau_fb = self.J * self.k_omega * (omega_des - omega)
        tau_des = tau_ff + tau_fb
        tau_des = (1.0 - self.k_tau_blend) * tau_des + self.k_tau_blend * tau

        # Желаемые токи
        I_des = _inverse_motors(T_des, tau_des, self.L_arm, self.k_F, self.I_min, self.I_max)

        I_des_dot = (I_des - self._I_des_prev) / dt
        self._I_des_prev = I_des.copy()

        # Вычисление Lyapunov-подобной функции
        I = np.array([I_L, I_R], dtype=float)
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

        # Backstepping на динамику токов
        I_cmd = I + self.tau_m * (I_des_dot - self.k_I * (I - I_des))
        I_cmd = np.clip(I_cmd, self.I_min, self.I_max)

        # Сохраняем желаемые значения для логирования
        self._v_des = v_des.copy()
        self._a_des = a_des.copy()
        self._theta_des = theta_des
        self._omega_des = omega_des
        self._T_des = T_des
        self._tau_des = tau_des
        self._I_des = I_des.copy()

        return I_cmd

    def get_desired(self):
        """Возвращает словарь желаемых величин, вычисленных на последнем шаге."""
        return {
            'v_des': self._v_des.copy(),
            'a_des': self._a_des.copy(),
            'theta_des': self._theta_des,
            'omega_des': self._omega_des,
            'T_des': self._T_des,
            'tau_des': self._tau_des,
            'I_des': self._I_des.copy(),
        }


class _VelocityMotorCascadeBase:
    """
    Каскад: желаемое ускорение -> (T_des, theta_des) -> внутренний ПД по тангажу -> инверсия моторов.
    Подклассы реализуют _accel_des.
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
        v_des = np.asarray(v_des, dtype=float).reshape(2)
        v = np.asarray(v, dtype=float).reshape(2)
        a = np.asarray(a, dtype=float).reshape(2)
        if dt <= 0.0:
            return np.clip(np.asarray(I, dtype=float).reshape(2), self.I_min, self.I_max)

        # Оценка тангажа по ускорению
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
    """Внешний контур: пропорционально ошибке скорости."""

    def __init__(self, m, g, J, L_arm, k_F, kp, k_th, k_w, a_lim, theta_lim, I_min, I_max):
        super().__init__(m, g, J, L_arm, k_F, k_th, k_w, a_lim, theta_lim, I_min, I_max)
        self.kp = float(kp)

    def _accel_des(self, v_des, v, a, dt):
        return self.kp * (v_des - v)


class PDVelocityMotorController(_VelocityMotorCascadeBase):
    """Внешний контур: ПД по скорости с использованием ускорения."""

    def __init__(self, m, g, J, L_arm, k_F, kp, kd, k_th, k_w, a_lim, theta_lim, I_min, I_max):
        super().__init__(m, g, J, L_arm, k_F, k_th, k_w, a_lim, theta_lim, I_min, I_max)
        self.kp = float(kp)
        self.kd = float(kd)

    def _accel_des(self, v_des, v, a, dt):
        e = v_des - v
        return self.kp * e - self.kd * a


class PIDVelocityMotorController(_VelocityMotorCascadeBase):
    """Внешний контур: ПИД по скорости."""

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