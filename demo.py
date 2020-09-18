# commento

import numpy as np
import matplotlib.pyplot as plt
from aisailib import GP, GM, skewgauss

# %% markdown

### initialize process and model

# %%

gp = GP()
gm = GM()

# %% markdown

### Define time and inner cause curve

#%%

stime = 100000
t = np.arange(stime)
inner_cause = skewgauss(n=stime, relative_location=0.5, alpha=4)

plt.plot(inner_cause)
# %% markdown

###   Loop through timesteps

# %%

da = 0
data = []
for t in range(stime):
    gm.mu_nui = inner_cause[t]
    gp.update(da)
    gp.generate()
    sp, ss = gp.sp, gp.ss
    da = gm.update((sp, ss))
    spg, ssg = gm.spg, gm.ssg
    os = gm.omega_s
    data.append((sp, ss, spg, ssg, os))

data = np.vstack(data)

# %% markdown

### Plot sensory anticipation vs sensory perceptions

# %%


sp, ss, spg, ssg, os = data.T

t = np.arange(len(ss))
plt.fill_between(t, ss - os, ss + os, color=[0.8, 0.8, 0.8])
p1, = plt.plot(t, sp, c='black', lw=1)
p2, = plt.plot(t, ss, c='black', lw=2)
p3, = plt.plot(t, spg, c='blue', lw=1)
p4, = plt.plot(t, ssg, c='blue', lw=2)
plt.legend([p1, p2, p3, p4], ['sp', 'ss', 'spg', 'ssg'])
