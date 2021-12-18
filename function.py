import numpy as np

import constant as C


def coth(x):
    y = np.cosh(x) / np.sinh(x)

    return y

def DB(x):
    y = 1/2 * (C.R**2 /(((C.R**2 + (C.L + x)**2))**(3/2)) - C.R**2 / ((x**2 + C.R**2)**(3/2)))

    return y

def V(x):
    y = 4/3 * C.PI * x**3

    return y

def B(x):
    y = 1/2 * ((C.L + x) / ((C.L + x)**2 + C.R**2)**(1/2) - x / (x**2 + C.R**2)**(1/2))

    return y

def Mu(x):
    m = 350000 * V(6.2 * 10**(-9)) * B(x) / (C.T * C.KB)
    
    # idx = np.argwhere(B(x) == 0)
    # assert not np.any(B(x) == 0), f"idx: {idx}, x: {x[idx]}, B(x): {B(x)[idx]}"
    
    y = 92 * 350000 * V(6.25 * 10**(-9)) * (coth(m) - 1/m)

    return y

# def F_M(x, n):
    # y = n* Mu(x) * DB(x)

    # return y

def F_M_vec(x, n):
    y = n * Mu(x) * DB(x)

    return y


# def D(X, i, j):
#     if i == j:
#         y = C.KB * C.T / (6 * C.PI * C.A * C.ETA)
#     else:
#         r_ij = X[j] - X[i]
#         y = 2 * C.KB * C.T / (8 * C.PI * C.ETA * r_ij)

#     return y

def D_vec(X, x_i):
    r_ij = X - x_i
    y = 2 * C.KB * C.T / (8 * C.PI * C.ETA * r_ij)

    return y


# def F_DD(X, N, i, j):
#     mu_0 = 4 * C.PI * 10**(-7)
#     mu_i = Mu(X[i]) * N[i]
#     mu_j = Mu(X[j]) * N[j]
#     y = 3 * mu_0 / (2 * C.PI) * mu_i * mu_j / (X[i] - X[j])**4
    
#     return y

def F_DD_vec(X, N, x_i, n_i):
    mu_0 = 4 * C.PI * 10**(-7)
    mu_i = Mu(x_i) * n_i
    mu = Mu(X) * N
    y = 3 * mu_0 / (2 * C.PI) * mu_i * mu / ((x_i - X)**3 * np.abs(x_i - X))
    
    return y

def GAMMA(n):
    if n == 1:
        y = 6 * C.PI * C.ETA * C.A * 2/3
    else:
        y = 6 * C.PI * C.ETA * C.A * 2/3 * n / (np.log(2 * n) - 0.5)
    
    return y