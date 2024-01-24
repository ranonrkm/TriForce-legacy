from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sympy import symbols, Eq, solve

def fake2real(fake,gamma=4):
    a = 1+ gamma*fake
    x = symbols('x')
    equation = Eq(x**(gamma+1) - a*x + a - 1, 0)
    solutions = solve(equation, x)
    return solutions[1]