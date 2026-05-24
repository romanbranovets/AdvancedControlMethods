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
    Backstepping-регулятор для планарного дрона, строго следующий выводу из README.

    Управление строится в отклонениях токов от висения:
        i_s = i_L + i_R   (коллективное),   i_d = i_R - i_L   (дифференциальное)
    Команды: u_s, u_d, которые затем переводятся в физические токи моторов.

    Вертикальная цепочка: u_s -> i_s -> v_z -> z
    Горизонтальная цепочка: u_d -> i_d -> omega -> theta -> v_x -> x
    """

    def __init__(
        self,
        m=0.5,
        g=9.81,
        J=0.014,
        L_arm=0.18,
        k_F=0.38,
        tau_m=0.07,
        # коэффициенты вертикального канала
        k_z1=0.8,
        k_z2=2.0,
        k_z3=8.0,
        # коэффициенты горизонтального канала
        k_x1=0.8,
        k_x2=1.5,
        k_x3=3.0,
        k_x4=5.0,
        k_x5=10.0,
        # ограничения
        theta_lim=0.42,   # рад
        I_min=0.0,
        I_max=16.0,
    ):
        self.m = float(m)
        self.g = float(g)
        self.J = float(J)
        self.L_arm = float(L_arm)
        self.k_F = float(k_F)
        self.tau_m = float(tau_m)

        # Ток висения и линеаризованные коэффициенты
        I0_sq = (m * g) / (2.0 * k_F)
        self.I0 = float(np.sqrt(max(I0_sq, 0.0)))
        self.a_s = 2.0 * k_F * self.I0 / m        # вертикальное ускорение / i_s
        self.a_d = 2.0 * L_arm * k_F * self.I0 / J  # угловое ускорение / i_d

        self.k_z1 = float(k_z1)
        self.k_z2 = float(k_z2)
        self.k_z3 = float(k_z3)

        self.k_x1 = float(k_x1)
        self.k_x2 = float(k_x2)
        self.k_x3 = float(k_x3)
        self.k_x4 = float(k_x4)
        self.k_x5 = float(k_x5)

        self.theta_lim = float(theta_lim)
        self.I_min = float(I_min)
        self.I_max = float(I_max)

        # память для вычисления производных виртуальных управлений
        self._alpha_z1_prev = 0.0
        self._alpha_z2_prev = 0.0
        self._alpha_x1_prev = 0.0
        self._alpha_x2_prev = 0.0
        self._alpha_x3_prev = 0.0
        self._alpha_x4_prev = 0.0

        # желаемые величины для логирования
        self._v_des = np.zeros(2, dtype=float)
        self._theta_des = 0.0
        self._omega_des = 0.0
        self._I_des = np.zeros(2, dtype=float)
        self._tau_des = 0.0

        self.last_lyapunov = None
        self.last_lyapunov_terms = None

    def reset(self, initial_state=None, target_pos=None):
        """
        Сброс внутреннего состояния контроллера.
        Если переданы initial_state и target_pos, вычисляются начальные значения
        виртуальных управлений, чтобы на первом шаге избежать больших скачков
        производных.
        """
        if initial_state is not None and target_pos is not None:
            self._init_previous_alphas(initial_state, target_pos)
        else:
            self._alpha_z1_prev = 0.0
            self._alpha_z2_prev = 0.0
            self._alpha_x1_prev = 0.0
            self._alpha_x2_prev = 0.0
            self._alpha_x3_prev = 0.0
            self._alpha_x4_prev = 0.0

        self._v_des[:] = 0.0
        self._theta_des = 0.0
        self._omega_des = 0.0
        self._I_des[:] = 0.0
        self._tau_des = 0.0

        self.last_lyapunov = None
        self.last_lyapunov_terms = None

    def _init_previous_alphas(self, state, target_pos):
        """
        Вычисляет виртуальные управления для начального состояния,
        принимая производные равными нулю, и сохраняет их как предыдущие.
        """
        x, z, vx, vz, theta, omega, I_L, I_R = state
        target_x, target_z = target_pos

        # Вертикальный канал
        e_z1 = z - target_z
        alpha_z1 = -self.k_z1 * e_z1
        e_z2 = vz - alpha_z1
        # dot{alpha_z1} = 0
        alpha_z2 = (0.0 - e_z1 - self.k_z2 * e_z2) / self.a_s

        # Горизонтальный канал
        e_x1 = x - target_x
        alpha_x1 = -self.k_x1 * e_x1
        e_x2 = vx - alpha_x1
        # dot{alpha_x1} = 0
        alpha_x2 = (0.0 - e_x1 - self.k_x2 * e_x2) / self.g
        alpha_x2 = np.clip(alpha_x2, -self.theta_lim, self.theta_lim)

        e_x3 = theta - alpha_x2
        # dot{alpha_x2} = 0
        alpha_x3 = 0.0 - self.g * e_x2 - self.k_x3 * e_x3

        e_x4 = omega - alpha_x3
        # dot{alpha_x3} = 0
        alpha_x4 = (0.0 - e_x3 - self.k_x4 * e_x4) / self.a_d

        self._alpha_z1_prev = alpha_z1
        self._alpha_z2_prev = alpha_z2
        self._alpha_x1_prev = alpha_x1
        self._alpha_x2_prev = alpha_x2
        self._alpha_x3_prev = alpha_x3
        self._alpha_x4_prev = alpha_x4

    def update(self, state, target_pos, a_meas, tau_meas, dt):
        """
        state: (8,) [x, z, vx, vz, theta, omega, I_L, I_R]
        target_pos: (2,) [x*, z*]
        a_meas, tau_meas: измеренные ускорение и момент (не используются в чистом backstepping)
        dt: шаг по времени
        """
        if dt <= 0.0:
            self.last_lyapunov = None
            self.last_lyapunov_terms = None
            return np.clip(state[6:8], self.I_min, self.I_max)

        x, z, vx, vz, theta, omega, I_L, I_R = state
        target_x, target_z = target_pos

        # Отклонения токов от висения
        i_L = I_L - self.I0
        i_R = I_R - self.I0
        i_s = i_L + i_R
        i_d = i_R - i_L

        # ------------------------------------------------------------------
        # Вертикальный канал (z, v_z, i_s)
        # ------------------------------------------------------------------
        e_z1 = z - target_z

        alpha_z1 = -self.k_z1 * e_z1
        alpha_z1_dot = (alpha_z1 - self._alpha_z1_prev) / dt
        self._alpha_z1_prev = alpha_z1

        e_z2 = vz - alpha_z1

        alpha_z2 = (alpha_z1_dot - e_z1 - self.k_z2 * e_z2) / self.a_s
        alpha_z2_dot = (alpha_z2 - self._alpha_z2_prev) / dt
        self._alpha_z2_prev = alpha_z2

        e_z3 = i_s - alpha_z2

        u_s = i_s + self.tau_m * (alpha_z2_dot - self.a_s * e_z2 - self.k_z3 * e_z3)

        # ------------------------------------------------------------------
        # Горизонтальный канал (x, v_x, theta, omega, i_d)
        # ------------------------------------------------------------------
        e_x1 = x - target_x

        alpha_x1 = -self.k_x1 * e_x1
        alpha_x1_dot = (alpha_x1 - self._alpha_x1_prev) / dt
        self._alpha_x1_prev = alpha_x1

        e_x2 = vx - alpha_x1

        alpha_x2 = (alpha_x1_dot - e_x1 - self.k_x2 * e_x2) / self.g
        alpha_x2 = np.clip(alpha_x2, -self.theta_lim, self.theta_lim)
        alpha_x2_dot = (alpha_x2 - self._alpha_x2_prev) / dt
        self._alpha_x2_prev = alpha_x2

        e_x3 = theta - alpha_x2

        alpha_x3 = alpha_x2_dot - self.g * e_x2 - self.k_x3 * e_x3
        alpha_x3_dot = (alpha_x3 - self._alpha_x3_prev) / dt
        self._alpha_x3_prev = alpha_x3

        e_x4 = omega - alpha_x3

        alpha_x4 = (alpha_x3_dot - e_x3 - self.k_x4 * e_x4) / self.a_d
        alpha_x4_dot = (alpha_x4 - self._alpha_x4_prev) / dt
        self._alpha_x4_prev = alpha_x4

        e_x5 = i_d - alpha_x4

        u_d = i_d + self.tau_m * (alpha_x4_dot - self.a_d * e_x4 - self.k_x5 * e_x5)

        # ------------------------------------------------------------------
        # Преобразование команд u_s, u_d в физические токи моторов
        # ------------------------------------------------------------------
        I_L_cmd = self.I0 + (u_s - u_d) / 2.0
        I_R_cmd = self.I0 + (u_s + u_d) / 2.0
        I_cmd = np.clip([I_L_cmd, I_R_cmd], self.I_min, self.I_max)

        # ------------------------------------------------------------------
        # Желаемые величины для логирования
        # ------------------------------------------------------------------
        # Желаемые скорости (виртуальные управления первого шага)
        v_des = np.array([alpha_x1, alpha_z1], dtype=float)
        theta_des = alpha_x2
        omega_des = alpha_x3
        i_s_des = alpha_z2
        i_d_des = alpha_x4

        # Желаемые физические токи моторов
        I_L_des = self.I0 + (i_s_des - i_d_des) / 2.0
        I_R_des = self.I0 + (i_s_des + i_d_des) / 2.0
        I_des = np.array([I_L_des, I_R_des], dtype=float)
        # Желаемый момент (из линеаризованного соотношения tau = J * a_d * i_d)
        tau_des = self.J * self.a_d * i_d_des

        # Сохраняем для get_desired()
        self._v_des = v_des
        self._theta_des = theta_des
        self._omega_des = omega_des
        self._I_des = I_des
        self._tau_des = tau_des

        # ------------------------------------------------------------------
        # Lyapunov-подобная функция
        # ------------------------------------------------------------------
        V_z = 0.5 * (e_z1**2 + e_z2**2 + e_z3**2)
        V_x = 0.5 * (e_x1**2 + e_x2**2 + e_x3**2 + e_x4**2 + e_x5**2)
        V_total = V_z + V_x

        # Компоненты для диагностического графика
        V_vel = 0.5 * (e_z2**2 + e_x2**2)          # ошибки скорости
        V_pitch = 0.5 * e_x3**2                    # ошибка угла тангажа
        V_omega = 0.5 * e_x4**2                    # ошибка угловой скорости
        V_current = 0.5 * (e_z3**2 + e_x5**2)      # ошибки токов

        self.last_lyapunov = float(V_total)
        self.last_lyapunov_terms = {
            'V_vel': float(V_vel),
            'V_pitch': float(V_pitch),
            'V_omega': float(V_omega),
            'V_current': float(V_current),
        }

        return I_cmd

    def get_desired(self):
        """Возвращает словарь желаемых величин, вычисленных на последнем шаге."""
        return {
            'v_des': self._v_des.copy(),
            'theta_des': self._theta_des,
            'omega_des': self._omega_des,
            'I_des': self._I_des.copy(),
            'tau_des': self._tau_des,
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
        self._theta_hat_prev = 0.0

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