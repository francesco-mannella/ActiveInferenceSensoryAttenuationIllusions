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

# # Dynamics

# The model latent variable has the following dynamics


# %% cell
display(syp.Eq(x.diff(t), h*x + a))
# %% markdown

# # Solution

# The solution of the equation above is given by:
# %%
f = syp.simplify(syp.dsolve(x.diff(t) + h*x - a, x))
f = syp.expand(f)
display(f)
# %% markdown

# # Derivative

# We can now derivate with respect to $a$:
# %%
dlhs = syp.Derivative("x(a)", a, evaluate=False)
drhs = syp.diff(f.rhs, a)
display(syp.Eq(dlhs, drhs))

# %%
