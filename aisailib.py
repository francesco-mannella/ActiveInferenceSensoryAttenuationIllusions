import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm
rng = np.random.RandomState()
# np.seterr(all='raise')


def skewgauss(n, relative_location=0, alpha=0):
    ''' Generate a squew gaussian.

        Args:
            n: (int) timesteps
            relative_location: (float) relative shifting of the
                distribution [0, 1] (0.1 indicates a shifting toward
                the beginning of the interval,0.9 indicates a shifting
                toward its end)
            alpha: (float) skewness

        Example:
            stime = 1500
            plt.plot(stime, skewgauss(stime, location=0.6, alpha=4))
    '''
    location = 10*relative_location - 5
    rng = np.array([-2, 2]) - location
    x = np.linspace(rng[0], rng[1], n)
    return skewnorm.pdf(x, alpha)


# %%
class GP:
    """ Generative process.

    Implementation of the generative process from the paper:

    Brown, H., Adams, R. A., Parees, I., Edwards, M., & Friston, K. (2013).
    Active inference, sensory attenuation and illusions. Cognitive Processing,
    14(4), 411–427. https://doi.org/10.1007/s10339-013-0571-3

    Attributes:
        pi_s: (float) Precision of sensory states probabilities.
        pi_x: (float) Precision of hidden state probabilities.
        h: (float) Integration step of hidden state dynamics.
        mu_xi: (float) Hidden state.
        dmu_xi: (float) Change of hidden state.
        mu_sp: (float) Proprioceptive sensory channel (central value).
        mu_ss: (float) Somatosensory sensory channel (central value).
        ve: (float) External cause.
        eta: (float) integration step
        omega_s: (float) standard deviation of sensory states
        omega_x: (float)  standard deviation of inner state increments
        a: (float) action
    """

    def __init__(self):

        self.pi_s = 8
        self.pi_x = 8

        self.h = 1.0/4.0

        self.mu_xi = 0
        self.dmu_xi = 0
        self.mu_sp = 0
        self.mu_ss = 0

        self.ve = 0.005

        self.eta = 0.00025

        self.omega_s = np.exp(-self.pi_s)
        self.omega_x = np.exp(-self.pi_x)

        self.a = 0

    def update(self, action_increment):
        """ Update dynamics of the process.

        Args:
            action_increment: (float) moves the current inner state.

        """

        da = action_increment
        self.a += self.eta*da

        self.mu_sp = self.mu_xi
        self.mu_ss = self.mu_xi + self.ve
        self.dmu_xi = np.tanh(self.a) - self.h*self.mu_xi

    def generate(self):
        """ Generate sensory states """

        self.sp = self.mu_sp
        self.ss = self.mu_ss
        self.dxi = self.dmu_xi
        self.mu_xi += self.eta*self.dxi


class GM:
    """ Generative Model.

    Attributes:
        pi_s: (float) Precision of sensory probabilities.
        pi_x: (float) Precision of hidden states probabilities.
        pi_nu: (float) Precision of hidden causes probabilities.
        h: (float) Integration step of hidden states dynamics.
        gamma: (float) Attenuation factor of sensory prediction error.
        mu_sp: (float) Proprioceptive sensory channel (central value).
        mu_ss: (float) Somatosensory sensory channel (central value).
        mu_xi: (float) Internal hidden state (central value).
        mu_xe: (float) External hidden state (central value).
        dmu_xi: (float) Change of internal hidden state (central value).
        dmu_xe: (float) Change of external hidden state (central value).
        mu_nui: (float) Internal cause (central value).
        mu_nue: (float) External cause (central value).
        da: (float) Increment of action
        eta: (float) Integration step
        omega_s: (float) Standard deviation of sensory states
        omega_x: (float)  Standard deviation of inner states
        omega_nu : (float) Standard deviation of inner causes

    """

    def __init__(self):

        self.pi_s_int = 8
        self.pi_x = 4
        self.pi_nu = 6

        self.h = 1.0/4.0
        self.gamma = 6

        self.mu_xi = 0
        self.mu_xe = 0
        self.dmu_xi = 0
        self.dmu_xe = 0
        self.mu_nui = 0
        self.mu_nue = 0

        self.da = 1/self.h
        self.eta = 0.00025

        self.omega_s = np.exp(-self.pi_s_int)
        self.omega_x = np.exp(-self.pi_x)
        self.omega_nu = np.exp(-self.pi_nu)

    def update(self, sensory_states):
        """ Update dynamics and give action

            Args:
                sensory_states: (float, float) current real proprioceptive and
                    somstosensory perception

            Returns:
                (float) current action increment
         """

        # update sensory states and dynamic precision
        self.sp, self.ss = sensory_states
        self.pi_s = self.pi_s_int - self.gamma*np.tanh(
            self.mu_xi + self.mu_nui)
        self.omega_s = np.exp(-self.pi_s)
        self.omega_x = np.exp(-self.pi_x)
        self.omega_nu = np.exp(-self.pi_nu)

        # dynamics of internal variables
        self.dmu_xi = self.mu_nui - self.h*self.mu_xi
        self.dmu_xe = self.mu_nue - self.h*self.mu_xe

        sp, ss = self.sp, self.ss
        os, ox = self.omega_s, self.omega_x
        mxi, mxe = self.mu_xi, self.mu_xe
        dmxi, dmxe = self.dmu_xi, self.dmu_xe
        ni, ne = self.mu_nui, self.mu_nue
        h, da = self.h, self.da

        # gradient descent optimization of mu_xi w.r.t. the free energy
        self.gd_mu_xi = \
            (1/os)*((sp + ss) - (mxe + 2 * mxi)) - \
            (1/ox)*h*((h + 1)*mxi - ni)
        # gradient descent optimization of mu_xe w.r.t. the free energy
        self.gd_mu_xe = \
            (1/os)*(ss - (mxe + mxi)) - \
            (1/ox)*h*((h + 1)*mxe - ne)
        # gradient descent optimization of dmu_xi w.r.t. the free energy
        self.gd_dmu_xi = -(1/ox)*(h*mxi + dmxi - ni)
        # gradient descent optimization of dmu_xe w.r.t. the free energy
        self.gd_dmu_xe = -(1/ox)*(h*mxe + dmxe - ne)
        # gradient descent optimization of action w.r.t. the free energy
        self.gd_a = (1/(os*da))*((2*mxi + mxe) - (sp + ss))

        # update with gradients
        self.dmu_xi += self.eta*self.gd_dmu_xi
        self.dmu_xe += self.eta*self.gd_dmu_xe
        self.mu_xi += self.eta*self.gd_mu_xi
        self.mu_xe += self.eta*self.gd_mu_xe

        self.spg = self.mu_xi
        self.ssg = self.mu_xi + self.mu_xe
        self.dxi = self.dmu_xi
        self.dxe = self.dmu_xe
        self.nui = self.mu_nui
        self.nue = self.mu_nue

        return self.gd_a


if __name__ == "__main__":

    gp = GP()
    gm = GM()

    # %%
    data = []

    stime = 100000
    t = np.arange(stime)
    ta = skewgauss(n=stime, relative_location=0.5, alpha=4)
    da = 0

    plt.plot(ta)
    # %%
    for t in range(stime):
        gm.mu_nui = ta[t]
        gp.update(da)
        gp.generate()
        sp, ss = gp.sp, gp.ss
        da = gm.update((sp, ss))
        spg, ssg = gm.spg, gm.ssg
        os = gm.omega_s
        data.append((sp, ss, spg, ssg, os))

    data = np.vstack(data)

    # %%
    sp, ss, spg, ssg, os = data.T

    t = np.arange(len(ss))
    plt.fill_between(t, ss - os, ss + os, color=[0.8, 0.8, 0.8])
    p1, = plt.plot(t, sp, c='black', lw=1)
    p2, = plt.plot(t, ss, c='black', lw=2)
    p3, = plt.plot(t, spg, c='blue', lw=1)
    p4, = plt.plot(t, ssg, c='blue', lw=2)
    plt.legend([p1, p2, p3, p4], ['sp', 'ss', 'spg', 'ssg'])
