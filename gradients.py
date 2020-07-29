
# %% cell

from IPython.display import display
import numpy as np
from matplotlib import pyplot as plt
import sympy as syp
from sympy import init_printing, symbols, sqrt, \
    exp, Inverse as inv, pi, log, re, Eq, diff
from sympy.matrices import Matrix, ones, zeros, eye
init_printing(use_unicode=True, use_latex='mathjax')


# %%

C = symbols("C", real="True")

sigma_s, sigma_x, sigma_nu = \
    symbols(r"\sigma_s \sigma_x \sigma_{\nu}", real=True)

s = Matrix(2, 1, symbols('s_p s_s'), real=True)
mux = Matrix(2, 1, symbols(r'\mu_{x_i} \mu_{x_e}'), real=True)
dmux = Matrix(2, 1, symbols(r'd\mu_{x_i} d\mu_{x_e}'), real=True)
munu = Matrix(2, 1, symbols(r'\mu_{\nu_i} \mu_{\nu_e}'), real=True)
dmunu = Matrix(2, 1, symbols(r'd\mu_{\nu_i} d\mu_{\nu_e}'), real=True)

Sigma_s = eye(2, real=True)*sigma_s
Sigma_x = eye(2, real=True)*sigma_x
Sigma_nu = eye(2, real=True)*sigma_nu


def g(x):
    W = eye(2, real=True)
    W[1, 0] = 1
    return W*x


def f(x, n):
    h = symbols("h", real=True, positive=True)
    return n - (eye(2, real=True)*h)*x


def normal(x, m, S):
    n = exp(-0.5*(x - m).T * inv(S) * (x - m)) \
        / sqrt(S.norm()*((2*pi)**2))
    return n


pF = normal(s, g(mux), Sigma_s) \
    * normal(dmux, f(mux, munu), Sigma_x)\
    * normal(dmunu, munu, Sigma_nu)

# %%
F = -log(pF[0]) - C
F = F.expand(force=True)
F = F.collect(Sigma_s)
F = F.collect(Sigma_x)
F = F.collect(Sigma_nu)
display(F)


# %%
d_mux = Eq(-diff("F", mux, evaluate=False),
           -diff(F, mux).simplify(), evaluate=False)
display(d_mux)

# %%
d_dmux = Eq(-diff("F", dmux, evaluate=False),
            -diff(F, dmux).simplify(), evaluate=False)
display(d_dmux)

# %%
d_dmunu = Eq(-diff("F", munu, evaluate=False),
             -diff(F, munu).simplify(), evaluate=False)
display(d_dmunu)
