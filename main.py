import numpy as np
import matplotlib.pyplot as plt
rng = np.random.RandomState()


class GP:
    """ Generative process.

    Implementation of the generative process from the paper:

    Brown, H., Adams, R. A., Parees, I., Edwards, M., & Friston, K. (2013).
    Active inference, sensory attenuation and illusions. Cognitive Processing,
    14(4), 411–427. https://doi.org/10.1007/s10339-013-0571-3

    Attributes:
        pi_s (float): Precision of sensory states probabilities.
        pi_x (float): Precision of hidden state probabilities.
        h (float): Integration step of hidden state dynamics.
        mu_xi (float): Hidden state.
        mu_dxi (float): Change of hidden state.
        mu_sp (float): Proprioceptive sensory channel (central value).
        mu_ss (float): Somatosensory sensory channel (central value).
        ve (float): External cause.

    """

    def __init__(self):

        self.pi_s = 8
        self.pi_x = 8

        self.h = 1.0/4.0

        self.mu_xi = 0
        self.mu_dxi = 0
        self.mu_sp = 0
        self.mu_ss = 0

        self.ve = 0.005

    def update(self, action):
        """Update dynamics of the process.

        Args:
            action (float): moves the current inner state.

        """

        a = action

        sigma_s = np.exp(-self.pi_s)
        sigma_x = np.exp(-self.pi_x)
        self.sp = self.mu_sp + sigma_s*rng.randn()
        self.ss = self.mu_ss + sigma_s*rng.randn()
        self.dxi = self.mu_dxi + sigma_x*rng.randn()

        self.mu_sp = self.mu_x,
        self.mu_ss = self.mu_x + self.ve
        self.mu_dxi = np.tanh(a) - self.h*self.mu_xi
        self.mu_xi += self.dxi


class GM:
    """Short summary.

    Attributes:
        pi_s (float): Precision of sensory probabilities.
        pi_x (float): Precision of hidden states probabilities.
        pi_nu (float): Precision of hidden causes probabilities.
        h (float): Integration step of hidden states dynamics.
        gamma (float): Attenuation factor of sensory prediction error.
        mu_sp (float): Proprioceptive sensory channel (central value).
        mu_ss (float): Somatosensory sensory channel (central value).
        mu_xi (float): Internal hidden state (central value).
        mu_xe (float): External hidden state (central value).
        mu_dxi (float): Change of internal hidden state (central value).
        mu_dxe (float): Change of external hidden state (central value).
        mu_nui (float): Internal cause (central value).
        mu_nue (float): External cause (central value).

    """

    def __init__(self):

        self.pi_s = 8
        self.pi_x = 4
        self.pi_nu = 6

        self.h = 1.0/4.0
        self.gamma = 0.1

        self.mu_sp = 0
        self.mu_ss = 0
        self.mu_xi = 0
        self.mu_xe = 0
        self.mu_dxi = 0
        self.mu_dxe = 0
        self.mu_nui = 0
        self.mu_nue = 0

    def update(self):

        self.pi_s = 8 - self.gamma*np.tanh(self.x[0] + self.nu[0])

        self.sigma_s = np.exp(-self.pi_s)
        self.sigma_x = np.exp(-self.pi_x)
        self.sigma_nu = np.exp(-self.pi_nu)

        self.sp = self.mu_sp + self.sigma_s*rng.randn()
        self.ss = self.mu_ss + self.sigma_s*rng.randn()
        self.dxi = self.mu_dxi + self.sigma_x*rng.randn()
        self.dxe = self.mu_dxe + self.sigma_x*rng.randn()
        self.nui = self.mu_nui + self.sigma_nu*rng.randn()
        self.nue = self.mu_nue + self.sigma_nu*rng.randn()

        self.mu_sp = self.mu_xi
        self.mu_ss = self.mu_xi + self.mu_xe
        self.mu_dxi = self.nui - self.h*self.mu_xi
        self.mu_dxe = self.nue - self.h*self.mu_xe

        self.mu_dxi = self.nui - self.h*self.mu_xi
        self.mu_dxe = self.nue - self.h*self.mu_xe

        self.xi += self.dxi
        self.xe += self.dxe


if __name__ == "__main__":
    pass
