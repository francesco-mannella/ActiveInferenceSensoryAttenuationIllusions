
# %% cell

from IPython.display import display
import numpy as np
from matplotlib import pyplot as plt
import sympy as syp
from sympy import init_printing, symbols, sqrt, \
    exp, Inverse as inv, pi, log, re, Eq, diff
from sympy.matrices import Matrix, ones, zeros, eye
init_printing(use_unicode=True)

# %% utlilities


def normal(x, m, S):
    """ normal equation """
    n = exp(-(x - m)**2/(2*S**2)) \
        / (S*sqrt(2*pi))
    return n

x,m,s =symbols("x m s")
syp.nsimplify(normal(x, m, x))


# %% Variables

sigma_s, sigma_x, sigma_nu = \
    symbols(r"\sigma_s \sigma_x \sigma_{\nu}", real=True)

s, mu_x, mu_nu = symbols(r's \mu_{x} \mu_{\nu}', real=True)
dmu_x = symbols(r"\mu'_{x}", real=True)
f = syp.Function(syp.Symbol("f", real=True))
g = syp.Function(syp.Symbol("g", real=True))

# %% VLE FRee Energy

p_s_mu = normal(s, g(mu_x), sigma_s)
p_mu = normal(dmu_x, f(mu_x, mu_nu), sigma_x)

F = -log(p_s_mu*p_mu)
F = F.collect(sigma_s)
F = F.collect(sigma_x)
F = F.collect(sigma_nu)

syp.nsimplify(F)
syp.simplify(F)


# %% gradient w.r.t. mu_x
gd_mu_x = -diff(F, mu_x)
gd_mu_x = syp.simplify(gd_mu_x)
gd_mu_x = syp.collect(gd_mu_x, sigma_x)
gd_mu_x = syp.collect(gd_mu_x, sigma_s)
gd_mu_x = syp.collect(gd_mu_x, diff(f(mu_x, mu_nu), mu_x))
gd_mu_x = syp.collect(gd_mu_x, diff(g(mu_x), mu_x))
gd_mu_x = syp.collect(gd_mu_x, 0.5)
gd_mu_x = syp.nsimplify(gd_mu_x)


gd_mu_x

# %% gradient w.r.t. dmu_x

gd_dmu_x = -diff(F, dmu_x)
gd_dmu_x = syp.simplify(gd_dmu_x)
gd_dmu_x = syp.collect(gd_dmu_x, sigma_x)
gd_dmu_x = syp.collect(gd_dmu_x, sigma_s)
gd_dmu_x = syp.collect(gd_dmu_x, diff(f(mu_x, mu_nu), mu_x))
gd_dmu_x = syp.collect(gd_dmu_x, diff(g(mu_x), mu_x))
gd_dmu_x = syp.collect(gd_dmu_x, 0.5)
gd_dmu_x = syp.nsimplify(gd_dmu_x)


gd_dmu_x

# %% gradient w.r.t. a
a = symbols("a")
ss = syp.Function("s")
F = F.subs(s, ss(a))
gd_a = -diff(F, a)

gd_a
