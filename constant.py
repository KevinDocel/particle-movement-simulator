import numpy as np
from scipy import constants as C

KB = C.Boltzmann
PI = C.pi

T = 295
ETA = 1.01 * 10**(-3)
A = 139.1 * 10**(-9)
L = 0.03
R = 0.0125


GAMMA = 6 * PI * ETA * A * 2/3 * 15 / (np.log(2 * 15) - 0.5)

D = KB * T / GAMMA