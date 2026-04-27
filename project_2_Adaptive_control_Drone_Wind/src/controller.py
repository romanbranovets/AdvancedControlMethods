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
    Cascaded PID baseline:
        outer  : pos error  -> desired velocity (clipped)
        middle : vel error  -> desired world acceleration
                              -> rotated into yaw-body frame
                              -> tilt commands + thrust
        inner  : attitude error -> body torques (PID)
        mixer  : (U1..U4) -> per-motor thrusts (X-config)

    Subclasses can override `_velocity_to_accel` or `_inner_loop`
    to swap a stage (e.g. with MRAC) without re-implementing the others.
    """

    def __init__(self,
                 m=0.5, g=9.81, l=0.2, d=0.01,
                 Ixx=0.01, Iyy=0.01, Izz=0.015,
                 max_tilt_deg=15.0,
                 max_speed_xy=2.5, max_speed_z=2.0,
                 max_accel_xy=8.0, max_accel_z=10.0,
                 thrust_ratio_min=0.2, thrust_ratio_max=3.0):
        self.m, self.g, self.l, self.d = m, g, l, d
        self.Ixx, self.Iyy, self.Izz = Ixx, Iyy, Izz
        self.max_tilt     = np.deg2rad(max_tilt_deg)
        self.max_speed_xy = max_speed_xy
        self.max_speed_z  = max_speed_z
        self.max_accel_xy = max_accel_xy
        self.max_accel_z  = max_accel_z
        self.thrust_min   = thrust_ratio_min * m * g
        self.thrust_max   = thrust_ratio_max * m * g

        self.pid_pos_x = PID(kp=1.2)
        self.pid_pos_y = PID(kp=1.2)
        self.pid_pos_z = PID(kp=1.5)

        self.pid_vel_x = PID(kp=2.5, ki=0.4, integral_limit=3.0)
        self.pid_vel_y = PID(kp=2.5, ki=0.4, integral_limit=3.0)
        self.pid_vel_z = PID(kp=4.0, ki=1.5, integral_limit=5.0)

        self.pid_roll  = PID(kp=10.0, ki=0.2, kd=2.0, integral_limit=1.0)
        self.pid_pitch = PID(kp=10.0, ki=0.2, kd=2.0, integral_limit=1.0)
        self.pid_yaw   = PID(kp=3.0,  ki=0.0, kd=0.5, integral_limit=1.0)

    def reset(self):
        for name in ['pid_pos_x', 'pid_pos_y', 'pid_pos_z',
                     'pid_vel_x', 'pid_vel_y', 'pid_vel_z',
                     'pid_roll',  'pid_pitch', 'pid_yaw']:
            getattr(self, name).reset()

    def update(self, state, target, dt, yaw_d=0.0):
        phi_d, theta_d, thrust_d = self._attitude_setpoint(state, target, dt)
        tau_roll, tau_pitch, tau_yaw = self._inner_loop(
            phi_d, theta_d, yaw_d, state, dt
        )
        return self._mix(thrust_d, tau_roll, tau_pitch, tau_yaw)

    def _attitude_setpoint(self, state, target, dt):
        pos = state[0:3]
        vel = state[3:6]
        psi = state[8]

        des_v = self._pos_to_vel(pos, target, dt)
        a_w   = self._velocity_to_accel(vel, des_v, dt)

        cpsi, spsi = np.cos(psi), np.sin(psi)
        ax_b =  cpsi * a_w[0] + spsi * a_w[1]
        ay_b = -spsi * a_w[0] + cpsi * a_w[1]

        denom = max(self.g + a_w[2], 0.3 * self.g)

        theta_d =  np.arctan2(ax_b, denom)
        phi_d   = -np.arctan2(ay_b, denom)
        theta_d = np.clip(theta_d, -self.max_tilt, self.max_tilt)
        phi_d   = np.clip(phi_d,   -self.max_tilt, self.max_tilt)

        cos_tilt = max(np.cos(phi_d) * np.cos(theta_d), 0.3)
        thrust_d = self.m * (self.g + a_w[2]) / cos_tilt
        thrust_d = np.clip(thrust_d, self.thrust_min, self.thrust_max)

        return phi_d, theta_d, thrust_d

    def _pos_to_vel(self, pos, target, dt):
        des_vx = self.pid_pos_x.update(target[0], pos[0], dt)
        des_vy = self.pid_pos_y.update(target[1], pos[1], dt)
        des_vz = self.pid_pos_z.update(target[2], pos[2], dt)
        des_vx = np.clip(des_vx, -self.max_speed_xy, self.max_speed_xy)
        des_vy = np.clip(des_vy, -self.max_speed_xy, self.max_speed_xy)
        des_vz = np.clip(des_vz, -self.max_speed_z,  self.max_speed_z)
        return np.array([des_vx, des_vy, des_vz])

    def _velocity_to_accel(self, vel, des_v, dt):
        ax = self.pid_vel_x.update(des_v[0], vel[0], dt)
        ay = self.pid_vel_y.update(des_v[1], vel[1], dt)
        az = self.pid_vel_z.update(des_v[2], vel[2], dt)
        ax = np.clip(ax, -self.max_accel_xy, self.max_accel_xy)
        ay = np.clip(ay, -self.max_accel_xy, self.max_accel_xy)
        az = np.clip(az, -self.max_accel_z,  self.max_accel_z)
        return np.array([ax, ay, az])

    def _inner_loop(self, phi_d, theta_d, yaw_d, state, dt):
        euler = state[6:9]
        tau_roll  = self.pid_roll.update(phi_d,   euler[0], dt)
        tau_pitch = self.pid_pitch.update(theta_d, euler[1], dt)
        tau_yaw   = self.pid_yaw.update(yaw_d,    euler[2], dt)
        return tau_roll, tau_pitch, tau_yaw

    def _mix(self, U1, U2, U3, U4):
        t1 = U1/4 - U3/(2*self.l) + U4/(4*self.d)
        t2 = U1/4 + U2/(2*self.l) - U4/(4*self.d)
        t3 = U1/4 + U3/(2*self.l) + U4/(4*self.d)
        t4 = U1/4 - U2/(2*self.l) - U4/(4*self.d)
        return np.maximum(np.array([t1, t2, t3, t4]), 0.0)


# ============================================================================
# MRAC with sigma-modification on velocity (translational) loop
# ============================================================================

class MRACAxis1D:
    """
    1st-order MRAC with sigma-modification for plant
        x_dot = u + Theta*^T Phi(x)            (matched, linear-in-parameters)

    Reference model (1st order, exponentially stable):
        x_m_dot = -a_m*(x_m - x_des)

    Control:
        u = a_m*(x_des - x) - Theta_hat^T Phi(x)

    Adaptive law (sigma-modification):
        Theta_hat_dot = gamma*e*Phi(x) - sigma*Theta_hat,    e = x - x_m

    Lyapunov function: V = 1/2 e^2 + 1/(2 gamma) ||Theta_tilde||^2
        V_dot <= -a_m e^2 - sigma ||Theta_tilde||^2 + sigma ||Theta*|| ||Theta_tilde||
                 -> uniformly ultimately bounded (UUB).

    Regressor Phi(x) options (n_basis):
        1: [1]                       constant disturbance only
        2: [1, x]                    + linear damping
        3: [1, x, |x|*x]   (default) + quadratic drag
    """

    def __init__(self, a_m=3.0, gamma=2.0, sigma=0.5,
                 theta_max=5.0, n_basis=3, u_max=15.0):
        self.a_m = a_m
        self.gamma = gamma
        self.sigma = sigma
        self.theta_max = theta_max
        self.n_basis = n_basis
        self.u_max = u_max
        self.theta = np.zeros(n_basis)
        self.x_m = 0.0

    def reset(self):
        self.theta[:] = 0.0
        self.x_m = 0.0

    def basis(self, x):
        if self.n_basis == 1:
            return np.array([1.0])
        if self.n_basis == 2:
            return np.array([1.0, x])
        return np.array([1.0, x, np.abs(x) * x])

    def update(self, x, x_des, dt):
        if dt <= 0.0:
            return 0.0

        decay = np.exp(-self.a_m * dt)
        self.x_m = x_des + (self.x_m - x_des) * decay

        e = x - self.x_m

        Phi = self.basis(x)
        u_nominal = self.a_m * (x_des - x)
        u_unsat   = u_nominal - self.theta @ Phi
        u = float(np.clip(u_unsat, -self.u_max, self.u_max))

        # adaptation freeze when saturated (avoids parameter drift due to
        # control authority loss, see Lavretsky & Wise §10.2)
        saturated = (u != u_unsat)
        if not saturated:
            dtheta = self.gamma * e * Phi - self.sigma * self.theta
            self.theta += dt * dtheta

            norm = np.linalg.norm(self.theta)
            if norm > self.theta_max:
                self.theta *= self.theta_max / norm

        return u


class MRACController(Controller):
    """
    Same outer (position-P) and inner (attitude-PID) loops as Controller,
    but the velocity loop is replaced by per-axis MRAC + sigma-modification.

    Plant for each translational axis:
        v_dot = u + Theta*^T Phi(v)

    where u is the commanded world acceleration and Theta*^T Phi captures
    aerodynamic drag + wind disturbance + modeling errors. The MRAC layer
    drives v -> v_m exponentially while online estimating Theta*.
    """

    def __init__(self,
                 m=0.5, g=9.81, l=0.2, d=0.01,
                 Ixx=0.01, Iyy=0.01, Izz=0.015,
                 a_m_xy=3.0, a_m_z=4.0,
                 gamma=2.0, sigma=0.5,
                 theta_max=5.0, n_basis=3,
                 log_history=True,
                 **kwargs):
        super().__init__(m=m, g=g, l=l, d=d,
                         Ixx=Ixx, Iyy=Iyy, Izz=Izz, **kwargs)

        self.mrac_vx = MRACAxis1D(a_m=a_m_xy, gamma=gamma, sigma=sigma,
                                  theta_max=theta_max, n_basis=n_basis,
                                  u_max=self.max_accel_xy)
        self.mrac_vy = MRACAxis1D(a_m=a_m_xy, gamma=gamma, sigma=sigma,
                                  theta_max=theta_max, n_basis=n_basis,
                                  u_max=self.max_accel_xy)
        self.mrac_vz = MRACAxis1D(a_m=a_m_z,  gamma=gamma, sigma=sigma,
                                  theta_max=theta_max, n_basis=n_basis,
                                  u_max=self.max_accel_z)

        self.log_history = log_history
        self.history = {
            't':      [],
            'v':      [],
            'v_m':    [],
            'v_des':  [],
            'theta':  [],   # (T, 3 axes, n_basis)
            'u':      [],
        }
        self._t_acc = 0.0

    def reset(self):
        super().reset()
        for ax in (self.mrac_vx, self.mrac_vy, self.mrac_vz):
            ax.reset()
        for k in self.history:
            self.history[k].clear()
        self._t_acc = 0.0

    def _velocity_to_accel(self, vel, des_v, dt):
        ux = self.mrac_vx.update(vel[0], des_v[0], dt)
        uy = self.mrac_vy.update(vel[1], des_v[1], dt)
        uz = self.mrac_vz.update(vel[2], des_v[2], dt)
        a = np.array([ux, uy, uz])

        if self.log_history:
            self.history['t'].append(self._t_acc)
            self.history['v'].append(vel.copy())
            self.history['v_m'].append(np.array([
                self.mrac_vx.x_m, self.mrac_vy.x_m, self.mrac_vz.x_m
            ]))
            self.history['v_des'].append(des_v.copy())
            self.history['theta'].append(np.array([
                self.mrac_vx.theta.copy(),
                self.mrac_vy.theta.copy(),
                self.mrac_vz.theta.copy(),
            ]))
            self.history['u'].append(a.copy())
            self._t_acc += dt

        return a

    def history_arrays(self):
        return {k: np.array(v) for k, v in self.history.items()}
