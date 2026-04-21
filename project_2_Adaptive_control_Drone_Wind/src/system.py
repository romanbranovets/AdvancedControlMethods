# src/system.py
import numpy as np

class QuadcopterSystem:
    def __init__(self):
        self.m = 0.5
        self.g = 9.81
        self.l = 0.2
        self.d = 0.01
        self.I = np.diag([0.01, 0.01, 0.015])

    def euler_to_rotation_matrix(self, euler):
        phi, theta, psi = euler
        cphi, sphi = np.cos(phi), np.sin(phi)
        cth, sth = np.cos(theta), np.sin(theta)
        cps, sps = np.cos(psi), np.sin(psi)

        R = np.array([
            [cps*cth,                  sps*cth,                 -sth],
            [cps*sth*sphi - sps*cphi, sps*sth*sphi + cps*cphi, cth*sphi],
            [cps*sth*cphi + sps*sphi, sps*sth*cphi - cps*sphi, cth*cphi]
        ])
        return R

    def dynamics(self, state, t, u, wind):
        pos = state[0:3]
        vel = state[3:6]
        euler = state[6:9].copy()
        omega = state[9:12]

        # === ЗАЩИТА ОТ СИНГУЛЯРНОСТИ ===
        theta = np.clip(euler[1], -np.pi/2 + 1e-3, np.pi/2 - 1e-3)
        euler[1] = theta

        R = self.euler_to_rotation_matrix(euler)

        thrust_body = np.array([0.0, 0.0, np.sum(u)])
        force_world = R @ thrust_body + np.array([0., 0., -self.m*self.g]) + wind
        acc = force_world / self.m

        # Моменты
        tau = np.zeros(3)
        tau[0] = self.l * (u[1] - u[3])          # roll
        tau[1] = self.l * (-u[0] + u[2])         # pitch
        tau[2] = self.d * (u[0] - u[1] + u[2] - u[3])  # yaw

        I_omega = self.I @ omega
        dot_omega = np.linalg.solve(self.I, tau - np.cross(omega, I_omega))

        # Кинематика Эйлера (безопасная)
        sphi, cphi = np.sin(euler[0]), np.cos(euler[0])
        cth = np.cos(euler[1])
        tth = np.tan(euler[1])

        phi_dot   = omega[0] + omega[1]*sphi*tth + omega[2]*cphi*tth
        theta_dot = omega[1]*cphi - omega[2]*sphi
        psi_dot   = (omega[1]*sphi + omega[2]*cphi) / cth

        euler_dot = np.array([phi_dot, theta_dot, psi_dot])

        dstate = np.concatenate((vel, acc, euler_dot, dot_omega))
        return dstate