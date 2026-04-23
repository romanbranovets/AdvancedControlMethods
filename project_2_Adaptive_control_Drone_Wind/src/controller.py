# src/controller.py
import numpy as np


class PID:
    """PID with derivative-on-measurement and integral clamping."""

    def __init__(self, kp=0.0, ki=0.0, kd=0.0, integral_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.integral = 0.0
        self.prev_meas = None

    def reset(self):
        self.integral = 0.0
        self.prev_meas = None

    def update(self, setpoint, measurement, dt):
        if dt <= 0.0:
            return 0.0
        error = setpoint - measurement
        self.integral += error * dt
        if self.integral_limit is not None:
            self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        if self.prev_meas is None:
            d_meas = 0.0
        else:
            d_meas = (measurement - self.prev_meas) / dt
        self.prev_meas = measurement
        return self.kp * error + self.ki * self.integral - self.kd * d_meas


class Controller:
    """
    Cascaded PID: position -> velocity -> acceleration -> attitude -> torque.

    Outer  loop: pos  error -> desired velocity (clipped)
    Middle loop: vel  error -> desired world-frame acceleration
                 -> rotated into yaw-body frame -> tilt commands + thrust
    Inner  loop: attitude error -> body torques
    Mixer: U1..U4 -> per-motor thrusts (X-configuration), with rate-limit on U1..U4.
    """

    def __init__(self,
                 m=0.5, g=9.81, l=0.2, d=0.01,
                 max_tilt_deg=15.0,
                 max_speed_xy=2.5, max_speed_z=2.0,
                 thrust_ratio_min=0.2, thrust_ratio_max=3.0):
        self.m, self.g, self.l, self.d = m, g, l, d
        self.max_tilt     = np.deg2rad(max_tilt_deg)
        self.max_speed_xy = max_speed_xy
        self.max_speed_z  = max_speed_z
        self.thrust_min   = thrust_ratio_min * m * g
        self.thrust_max   = thrust_ratio_max * m * g

        # outer: position -> velocity setpoint
        self.pid_pos_x = PID(kp=1.2, ki=0.0, kd=0.0)
        self.pid_pos_y = PID(kp=1.2, ki=0.0, kd=0.0)
        self.pid_pos_z = PID(kp=1.5, ki=0.0, kd=0.0)

        # middle: velocity -> desired acceleration [m/s^2]
        self.pid_vel_x = PID(kp=2.5, ki=0.4, kd=0.0, integral_limit=3.0)
        self.pid_vel_y = PID(kp=2.5, ki=0.4, kd=0.0, integral_limit=3.0)
        self.pid_vel_z = PID(kp=4.0, ki=1.5, kd=0.0, integral_limit=5.0)

        # inner: attitude -> torque
        self.pid_roll  = PID(kp=10.0, ki=0.2, kd=2.0, integral_limit=1.0)
        self.pid_pitch = PID(kp=10.0, ki=0.2, kd=2.0, integral_limit=1.0)
        self.pid_yaw   = PID(kp=3.0,  ki=0.0, kd=0.5, integral_limit=1.0)

    def reset(self):
        for name in ['pid_pos_x', 'pid_pos_y', 'pid_pos_z',
                     'pid_vel_x', 'pid_vel_y', 'pid_vel_z',
                     'pid_roll',  'pid_pitch', 'pid_yaw']:
            getattr(self, name).reset()

    def update(self, state, target, dt, yaw_d=0.0):
        pos   = state[0:3]
        vel   = state[3:6]
        euler = state[6:9]
        psi = euler[2]

        # ---- outer: position -> desired velocity ----
        des_vx = self.pid_pos_x.update(target[0], pos[0], dt)
        des_vy = self.pid_pos_y.update(target[1], pos[1], dt)
        des_vz = self.pid_pos_z.update(target[2], pos[2], dt)

        des_vx = np.clip(des_vx, -self.max_speed_xy, self.max_speed_xy)
        des_vy = np.clip(des_vy, -self.max_speed_xy, self.max_speed_xy)
        des_vz = np.clip(des_vz, -self.max_speed_z,  self.max_speed_z)

        # ---- middle: velocity -> desired world acceleration ----
        ax_w = self.pid_vel_x.update(des_vx, vel[0], dt)
        ay_w = self.pid_vel_y.update(des_vy, vel[1], dt)
        az_w = self.pid_vel_z.update(des_vz, vel[2], dt)

        # rotate horizontal accel into yaw-aligned frame
        cpsi, spsi = np.cos(psi), np.sin(psi)
        ax_b =  cpsi * ax_w + spsi * ay_w
        ay_b = -spsi * ax_w + cpsi * ay_w

        # keep vertical divisor strictly positive
        denom = max(self.g + az_w, 0.3 * self.g)

        # tilt commands (ZYX, R_b2w -> thrust_world = T*[s_theta, -s_phi, c_theta*c_phi] for psi=0)
        theta_d =  np.arctan2(ax_b, denom)
        phi_d   = -np.arctan2(ay_b, denom)

        theta_d = np.clip(theta_d, -self.max_tilt, self.max_tilt)
        phi_d   = np.clip(phi_d,   -self.max_tilt, self.max_tilt)

        # thrust magnitude with gravity/tilt compensation
        cos_tilt = max(np.cos(phi_d) * np.cos(theta_d), 0.3)
        thrust_d = self.m * (self.g + az_w) / cos_tilt
        thrust_d = np.clip(thrust_d, self.thrust_min, self.thrust_max)

        # ---- inner: attitude -> torques ----
        tau_roll  = self.pid_roll.update(phi_d,   euler[0], dt)
        tau_pitch = self.pid_pitch.update(theta_d, euler[1], dt)
        tau_yaw   = self.pid_yaw.update(yaw_d,    euler[2], dt)

        U1, U2, U3, U4 = thrust_d, tau_roll, tau_pitch, tau_yaw

        # X-configuration mixer
        t1 = U1/4 - U3/(2*self.l) + U4/(4*self.d)
        t2 = U1/4 + U2/(2*self.l) - U4/(4*self.d)
        t3 = U1/4 + U3/(2*self.l) + U4/(4*self.d)
        t4 = U1/4 - U2/(2*self.l) - U4/(4*self.d)

        u = np.maximum(np.array([t1, t2, t3, t4]), 0.0)
        return u
