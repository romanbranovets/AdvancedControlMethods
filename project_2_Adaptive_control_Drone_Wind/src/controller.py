# src/controller.py
import numpy as np

class PID:
    def __init__(self, kp=0.0, ki=0.0, kd=0.0, integral_limit=20.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.integral_limit = integral_limit

    def update(self, error, dt):
        if dt <= 0:
            return 0.0
        self.integral = np.clip(self.integral + error * dt, -self.integral_limit, self.integral_limit)
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


class Controller:
    """
    Каскадный PID с ограничением наклона 10° и плавным управлением.
    """
    def __init__(self):
        self.m = 0.5
        self.g = 9.81
        self.l = 0.2
        self.d = 0.01

        # === ВНЕШНИЙ КОНТУР: позиция → желаемая скорость (медленная) ===
        self.pid_pos_x = PID(kp=0.05, ki=0.0, kd=0.0)
        self.pid_pos_y = PID(kp=0.05, ki=0.0, kd=0.0)
        self.pid_pos_z = PID(kp=0.15, ki=0.0, kd=0.0)

        # === СРЕДНИЙ КОНТУР: скорость → желаемый угол / добавка тяги ===
        self.pid_vel_x = PID(kp=0.2, ki=0.0, kd=0.15)
        self.pid_vel_y = PID(kp=0.2, ki=0.0, kd=0.15)
        self.pid_vel_z = PID(kp=0.8, ki=0.0, kd=0.4)

        # === ВНУТРЕННИЙ КОНТУР (быстрый, но с ограниченным воздействием) ===
        self.pid_roll  = PID(kp=15.0, ki=0.5, kd=4.0)
        self.pid_pitch = PID(kp=15.0, ki=0.5, kd=4.0)
        self.pid_yaw   = PID(kp=2.0,  ki=0.1, kd=1.0)

        # Для плавного изменения моторов
        self.prev_u = np.ones(4) * (self.m * self.g / 4)  # начальная тяга висения
        self.max_slew_rate = 15.0  # максимальное изменение тяги за секунду

    def update(self, state, target, dt):
        pos = state[0:3]
        vel = state[3:6]
        euler = state[6:9]

        err_pos_x = target[0] - pos[0]
        err_pos_y = target[1] - pos[1]
        err_pos_z = target[2] - pos[2]

        # Position → desired velocity (сильно ограничена)
        des_vel_x = self.pid_pos_x.update(err_pos_x, dt)
        des_vel_y = self.pid_pos_y.update(err_pos_y, dt)
        des_vel_z = self.pid_pos_z.update(err_pos_z, dt)

        # Ограничение желаемой скорости (медленное движение)
        max_speed_xy = 1.0   # м/с (ещё медленнее)
        max_speed_z  = 0.8   # м/с
        des_vel_x = np.clip(des_vel_x, -max_speed_xy, max_speed_xy)
        des_vel_y = np.clip(des_vel_y, -max_speed_xy, max_speed_xy)
        des_vel_z = np.clip(des_vel_z, -max_speed_z,  max_speed_z)

        # Velocity → desired angle / thrust
        # ВАЖНО: знак для тангажа инвертирован!
        theta_d = -self.pid_vel_x.update(des_vel_x - vel[0], dt)
        phi_d   =  self.pid_vel_y.update(des_vel_y - vel[1], dt)
        thrust_add = self.pid_vel_z.update(des_vel_z - vel[2], dt)

        thrust_d = self.m * self.g + thrust_add

        # Ограничения тяги и углов (углы не более 10°)
        thrust_d = np.clip(thrust_d, self.m * self.g * 0.5, self.m * self.g * 3.5)
        max_tilt = np.deg2rad(10)
        phi_d   = np.clip(phi_d,   -max_tilt, max_tilt)
        theta_d = np.clip(theta_d, -max_tilt, max_tilt)

        # Внутренний контур
        tau_roll  = self.pid_roll.update(phi_d - euler[0], dt)
        tau_pitch = self.pid_pitch.update(theta_d - euler[1], dt)
        tau_yaw   = self.pid_yaw.update(0.0 - euler[2], dt)

        U1 = thrust_d
        U2 = tau_roll
        U3 = tau_pitch
        U4 = tau_yaw

        t1 = U1/4 - U3/(2*self.l) + U4/(4*self.d)
        t2 = U1/4 + U2/(2*self.l) - U4/(4*self.d)
        t3 = U1/4 + U3/(2*self.l) + U4/(4*self.d)
        t4 = U1/4 - U2/(2*self.l) - U4/(4*self.d)

        u_raw = np.maximum([t1, t2, t3, t4], 0.0)

        # Плавное изменение скоростей моторов (rate limiter)
        max_change = self.max_slew_rate * dt
        u_smooth = np.zeros(4)
        for i in range(4):
            diff = u_raw[i] - self.prev_u[i]
            diff = np.clip(diff, -max_change, max_change)
            u_smooth[i] = self.prev_u[i] + diff

        self.prev_u = u_smooth
        return u_smooth