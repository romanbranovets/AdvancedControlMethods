# src/system.py
import numpy as np


class QuadcopterSystem:
    """
    12-state quadrotor with ZYX Euler parameterization.

    state = [x, y, z,  vx, vy, vz,  phi, theta, psi,  p, q, r]
             pos         vel           euler             body rates

    Inputs:
        u       : np.ndarray(4), per-motor thrusts [N], clipped to [u_min, u_max]
        v_wind  : np.ndarray(3), wind velocity in world frame [m/s]

    Forces:
        thrust  : R_b2w @ [0,0, sum(u)]
        drag    : linear + quadratic on v_rel = v - v_wind
        gravity : [0, 0, -m g]

    Moments (body frame, X-configuration):
        tau_x = l*(u1 - u3)
        tau_y = l*(-u0 + u2)
        tau_z = d*(u0 - u1 + u2 - u3)
    """

    def __init__(self,
                 m=0.5, g=9.81,
                 l=0.2, d=0.01,
                 Ixx=0.01, Iyy=0.01, Izz=0.015,
                 c_drag_lin=0.15, c_drag_quad=0.05,
                 u_min=0.0, u_max=5.0):
        self.m = m
        self.g = g
        self.l = l
        self.d = d
        self.I = np.diag([Ixx, Iyy, Izz])
        self.c_drag_lin = c_drag_lin
        self.c_drag_quad = c_drag_quad
        self.u_min = u_min
        self.u_max = u_max

    @staticmethod
    def R_body_to_world(euler):
        # R = Rz(psi) @ Ry(theta) @ Rx(phi)
        phi, theta, psi = euler
        cphi, sphi = np.cos(phi), np.sin(phi)
        cth,  sth  = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi),  np.sin(psi)
        R = np.array([
            [cpsi*cth,  cpsi*sth*sphi - spsi*cphi,  cpsi*sth*cphi + spsi*sphi],
            [spsi*cth,  spsi*sth*sphi + cpsi*cphi,  spsi*sth*cphi - cpsi*sphi],
            [-sth,      cth*sphi,                    cth*cphi]
        ])
        return R

    def dynamics(self, state, t, u, v_wind):
        u = np.clip(np.asarray(u, dtype=float), self.u_min, self.u_max)

        vel   = state[3:6]
        euler = state[6:9].copy()
        omega = state[9:12]

        # gimbal-lock guard on pitch
        euler[1] = np.clip(euler[1], -np.pi/2 + 1e-3, np.pi/2 - 1e-3)

        R = self.R_body_to_world(euler)

        # --- forces (world frame) ---
        thrust_world = R @ np.array([0.0, 0.0, np.sum(u)])

        v_rel = vel - np.asarray(v_wind, dtype=float)
        v_rel_norm = np.linalg.norm(v_rel)
        drag = -self.c_drag_lin * v_rel - self.c_drag_quad * v_rel_norm * v_rel

        grav = np.array([0.0, 0.0, -self.m * self.g])

        acc = (thrust_world + drag + grav) / self.m

        # --- torques (body frame) ---
        tau = np.array([
            self.l * (u[1] - u[3]),
            self.l * (-u[0] + u[2]),
            self.d * (u[0] - u[1] + u[2] - u[3]),
        ])

        I_omega = self.I @ omega
        dot_omega = np.linalg.solve(self.I, tau - np.cross(omega, I_omega))

        # --- Euler kinematics (body rates -> euler rates) ---
        sphi, cphi = np.sin(euler[0]), np.cos(euler[0])
        cth = np.cos(euler[1])
        tth = np.tan(euler[1])

        phi_dot   = omega[0] + omega[1]*sphi*tth + omega[2]*cphi*tth
        theta_dot = omega[1]*cphi - omega[2]*sphi
        psi_dot   = (omega[1]*sphi + omega[2]*cphi) / cth

        euler_dot = np.array([phi_dot, theta_dot, psi_dot])

        return np.concatenate([vel, acc, euler_dot, dot_omega])
