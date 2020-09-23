import numpy as np
import matplotlib.pyplot as plt
from aisailib import GP, GM, skewgauss

# %% markdown

# ### initialize process and model

# %%

gp = GP()
gm = GM()

# %% markdown

# ### Define time and inner cause curve

# %%

stime = 100000
da = 0
t = np.arange(stime)
inner_cause = skewgauss(n=stime, relative_location=0.5, alpha=4)

plt.plot(inner_cause)

# %% markdown

# ###   Loop through timesteps

# %%

data = []
for gamma in [2, 4, 6, 8]:

    data_gamma = []
    gp = GP()
    gm = GM()
    gm.gamma = gamma
    for t in range(stime):
        gm.mu_nui = inner_cause[t]
        gp.update(da)
        gp.generate()
        sp, ss = gp.sp, gp.ss
        da = gm.update((sp, ss))
        spg, ssg = gm.spg, gm.ssg
        os = gm.omega_s
        ps = gm.pi_s
        data_gamma.append((sp, ss, spg, ssg, os, ps))
    data_gamma = np.vstack(data_gamma)
    data.append(data_gamma)

# %% markdown

# ### Plot sensory anticipation vs sensory perceptions

# %%
plt.figure(figsize=(12, 3))
for i, gamma in enumerate([2, 4, 6, 8]):
    sp, ss, spg, ssg, os, ps = data[i].T
    plt.subplot(1, 4, i+1)
    t = np.arange(len(ss))
    plt.fill_between(t, ss - os, ss + os, color=[0.8, 0.8, 0.8])
    p1, = plt.plot(t, sp, c='black', lw=1)
    p2, = plt.plot(t, ss, c='black', lw=2)
    p3, = plt.plot(t, spg, c='blue', lw=1)
    p4, = plt.plot(t, ssg, c='blue', lw=2)
    plt.legend([p1, p2, p3, p4], ['sp', 'ss', 'spg', 'ssg'])
