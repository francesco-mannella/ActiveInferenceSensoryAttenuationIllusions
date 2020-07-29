# %% cell

from IPython.display import display
import numpy as np
from matplotlib import pyplot as plt
import sympy as syp
from sympy.physics.mechanics import dynamicsymbols, init_vprinting
init_vprinting()

# %% cell

x = dynamicsymbols('x')
n, t, a = syp.symbols("n t a")
h = syp.Symbol("h", positive=True)

# %% markdown

# # Dynamics, its solution and derivative w.r.t. a

# %% cell
display(syp.Eq(x.diff(t), n - h*x + a))
f = syp.simplify(syp.dsolve(x.diff(t) - n + h*x - a, x))
f = syp.expand(f)
display(f)
dlhs = syp.Derivative("x(a)", a, evaluate=False)
drhs = syp.diff(f.rhs, a)
display(syp.Eq(dlhs, drhs))

# %%
