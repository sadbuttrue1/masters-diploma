from sympy import *
import numpy as np
from scipy.constants import epsilon_0

k_c = (4 * np.pi * epsilon_0) ** (-1)


def build_cm(R, d, n):
    Cm = zeros(n, n)
    for i in range(n):
        for j in range(n):
            if i == j:
                Cm[i, j] = 1 / R[i]
            else:
                Cm[i, j] = 1 / d[i, j]
    return Cm


def find_charges(R, d, n, Phi):
    Cm = build_cm(R, d, n)
    Cmi = Cm ** -1
    Cmi.simplify()
    q = k_c * Cmi * Phi
    return q
