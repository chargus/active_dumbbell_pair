import numpy as np

"""
Run MD simulation for a simple 2-dumbbell system entirely in Python.
"""


class Simulator(object):

    def __init__(self, pos0, dt=0.001, factive=0.0,
                 sigma=1.0, epsilon=1.0, k=100.0, r0=1.0):
        self.pos = pos0
        self.vel = np.zeros_like(self.pos)
        self.dt = dt
        self.factive = factive
        self.sigma = sigma
        self.epsilon = epsilon
        self.k = k
        self.r0 = r0

    def lj(self, r2):
        r12 = (self.sigma**2 / r2)**6
        r6 = (self.sigma**2 / r2)**3
        f = (24 / r2) * self.epsilon * (2 * r12 - r6)
        return f

    def bond_harm(self, r2):
        return -2 * self.k * (np.sqrt(r2) - self.r0)

    def active_torque(self, dx, dy):
        fx = np.empty(4)
        fy = np.empty(4)
        dumbbell_ids = [(0, 1), (2, 3)]
        for i1, i2 in dumbbell_ids:
            delx = dx[i2, i1]
            dely = dy[i2, i1]
            rsq = delx * delx + dely * dely
            r = np.sqrt(rsq)
            delx /= r
            dely /= r
            fx[i1] = self.factive * (dely)
            fy[i1] = self.factive * (-delx)
            fx[i2] = self.factive * (-dely)
            fy[i2] = self.factive * (delx)
        return fx, fy

    def get_forces(self):
        """Compute force vector on each particle.
        """
        dx = np.subtract.outer(self.pos[:, 0], self.pos[:, 0])
        dy = np.subtract.outer(self.pos[:, 1], self.pos[:, 1])
        r2 = dx**2 + dy**2  # Squared distance between all particle pairs

        # Define masks governing interaction types
        nonbond = np.array([[0, 0, 1, 1],
                            [0, 0, 1, 1],
                            [1, 1, 0, 0],
                            [1, 1, 0, 0]]).astype(bool)

        bond = np.array([[0, 1, 0, 0],
                         [1, 0, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]]).astype(bool)

        # Compute forces
        fx = np.zeros_like(dx)
        fy = np.zeros_like(dy)
        lj_force = self.lj(r2[nonbond])
        spring_force = self.bond_harm(r2[bond])
        fx[nonbond] = -dx[nonbond] * lj_force
        fy[nonbond] = -dy[nonbond] * lj_force
        fx[bond] = -dx[bond] * spring_force
        fy[bond] = -dy[bond] * spring_force
        net_fx = np.sum(fx, axis=0)
        net_fy = np.sum(fy, axis=0)
        active_fx, active_fy = self.active_torque(dx, dy)
        forces = np.stack([net_fx + active_fx, net_fy + active_fy], axis=1)
        return forces

    def velocity_verlet_timestep(self):
        """Advance positions and velocities one timestep using Velocity Verlet
        """
        f1 = self.get_forces()
        self.vel = self.vel + 0.5 * self.dt * f1
        self.pos = self.pos + self.dt * self.vel
        f2 = self.get_forces()
        self.vel = self.vel + 0.5 * self.dt * f2

    def enforce_constraints(self, f):
        """Remove translation and stretch part of force
        """
        f_removed0 = np.zeros(2)
        f_removed1 = np.zeros(2)
        f_removed0 = 0.5 * (f[0] + f[1])
        f_removed1 = 0.5 * (f[2] + f[3])
        f[:2] -= f_removed0
        f[2:] -= f_removed1
        return f_removed1

    def impose_bath(self, T, gamma):
        vtemp = self.vel.copy()
        ncoeff = np.sqrt(T * gamma * self.dt / 8.)
        rand = np.random.normal(size=(4, 2))
        for i in (0, 1):  # First x, then y
            # Fluctuations
            self.vel[0, i] += ncoeff * (rand[0, i] - rand[1, i])
            self.vel[1, i] += ncoeff * (rand[1, i] - rand[0, i])
            self.vel[2, i] += ncoeff * (rand[2, i] - rand[3, i])
            self.vel[3, i] += ncoeff * (rand[3, i] - rand[2, i])

            # Dissipation
            self.vel[0, i] -= .25 * gamma * (vtemp[0, i] - vtemp[1, i])
            self.vel[1, i] -= .25 * gamma * (vtemp[1, i] - vtemp[0, i])
            self.vel[2, i] -= .25 * gamma * (vtemp[2, i] - vtemp[3, i])
            self.vel[3, i] -= .25 * gamma * (vtemp[3, i] - vtemp[2, i])

    def vv_constrained(self, T=1.0, gamma=0.0):
        """Velocity Verlet timestep with pinned COM and rigid rotors

        Parameters
        ----------
        T : float, optional
            Langevin bath temperature
        gamma : float, optional
            Rotational friction factor

        """
        f_removed = np.zeros(2)

        self.impose_bath(T, gamma)
        f = self.get_forces()
        f_removed += self.enforce_constraints(f)
        self.vel = self.vel + 0.5 * self.dt * f
        self.pos = self.pos + self.dt * self.vel
        f = self.get_forces()
        f_removed += self.enforce_constraints(f)
        self.vel = self.vel + 0.5 * self.dt * f
        self.impose_bath(T, gamma)
        return f_removed
