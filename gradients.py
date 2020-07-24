
# %%

from IPython.display import display, Latex
import numpy as np
from matplotlib import pyplot as plt
import sympy as syp
from sympy import init_printing
from sympy.matrices import Matrix, ones, eye, Inverse as inv
from sympy.functions import transpose as t
from IPython.display import display, Latex

init_printing(use_unicode=True, use_latex='mathjax')


# %%


def normal(x, mu, sig):
    """Give normal gaussian equation.

    Args:
        x (Matrix 2x1): indipendent variable.
        mu (Matrix 2x1): Mean.
        sig (Matrix 2x2): Standard deviation.

    Returns:
        float or array: normal function of each value in x

    """
    return syp.exp(-0.5*(x - mu).T*syg.Inverse(sig)*(x - mu))\
        / syp.sqrt(syp.norm(sig)*(2*syp.pi)**2)
