import numpy as np


# k here is k+1 in the paper
def F(a, x, lambd, rho, nu):
    return 0.5 * (a ** 2 - 1j * a) + lambd * (1j * a * rho * nu - 1) * x + 0.5 * (lambd * nu * x) ** 2


def h_pre_estim(a, k, delta, alpha, lambd, rho, nu, memory):
    j = np.arange(k)
    tj = delta * j
    b = (delta ** (alpha + 1) / np.euler_gamma(alpha + 1)) * ((k - j) ** alpha - (k - 1 - j) ** alpha)
    h_estim = np.vectorize(lambda t: h_numerical(a, t, delta, alpha, lambd, rho, nu, memory))(tj)
    return np.sum(b * F(a, h_estim, lambd, rho, nu))


def h_numerical(a, k, delta, alpha, lambd, rho, nu, memory):
    if k == 0:
        return 0
    if k in memory.keys():
        return memory[k]
    j = np.arange(k)
    arr_h_pre_estim = np.vectorize(lambda t: h_pre_estim(a, t, delta, alpha, lambd, rho, nu, memory))(j)
    a0 = (delta ** (alpha + 1) / np.euler_gamma(alpha + 2)) * \
         ((k - 1) ** (alpha + 1) - (k - 1 - alpha) * k ** alpha)
    aj = (delta ** (alpha + 1) / np.euler_gamma(alpha + 2)) * \
        ((k - j + 1) ** (alpha + 1) - (k - 1 - j) ** (alpha + 1) - 2 * (k - j) ** (alpha + 1))
    arr_a = np.append([a0], aj)
    arr_f_h_estim = np.array([F(a, h_numerical(a, i, delta, alpha, lambd, rho, nu, memory), lambd, rho, nu) for i in j])
    hk = np.sum(arr_a[:-1] * arr_f_h_estim + a[-1] * F(a, arr_h_pre_estim, lambd, rho, nu))
    memory[k] = hk
    return hk


# TODO : estimate fractional integrals on h
def Lp(theta, v0, a, t, delta, alpha, lambd, rho, nu, memory):
    # approximate the fractional integral with a riemannian sum
    memory = {}
    n = 100
    k = np.arange(1, n + 1)
    hk = h_numerical(a, k, delta, alpha, lambd, rho, nu, memory)


