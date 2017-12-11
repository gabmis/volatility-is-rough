import numpy as np
from math import e, pow, gamma

# k here is k+1 in the paper
def F(a, x, lambd, rho, nu):
    return 0.5 * (a ** 2 - 1j * a) + lambd * (1j * a * rho * nu - 1) * x + 0.5 * (lambd * nu * x) ** 2

def a_(j,kplus1, alpha, delta):
    if j == 0:
        return delta**alpha/gamma(alpha+2)*((kplus1-1)**alpha+1-(kplus1-1-alpha)*kplus1**alpha)
    elif j<kplus1:
        return delta**alpha/gamma(alpha+2)*((kplus1+1-j)**(alpha+1) + (kplus1-1-j)**(alpha+1)-2*(kplus1-j)**(alpha+1))
    else :
        return delta**alpha/gamma(alpha+2)

def b_(j, kplus1, alpha, delta):
    return delta**alpha/gamma(alpha+1)*((kplus1-j)**alpha-(kplus1-1-j)**alpha)

def h_pre_estim(a, k, delta, alpha, lambd, rho, nu, memory, memory_pre):
    if k == 0:
        return 0
    if k in memory_pre.keys():
        return memory_pre[k]
    j = np.array(k)
    b = np.vectorize(lambda t: b_(t, k, alpha, delta))(j)
    #b = (delta ** (alpha + 1) / gamma(alpha + 1)) * ((k - j) ** alpha - (k - 1 - j) ** alpha)
    h_estim = np.vectorize(lambda t: h_numerical(a, t, delta, alpha, lambd, rho, nu, memory, memory_pre))(j)
    hk_pre = np.sum(b * F(a, h_estim, lambd, rho, nu))
    memory_pre[k]=hk_pre
    return hk_pre


def h_numerical(a, k, delta, alpha, lambd, rho, nu, memory, memory_pre):
    if k == 0:
        return 0
    if k in memory.keys():
        return memory[k]
    j = np.arange(k)
    arr_h_pre_estim = np.vectorize(lambda t: h_pre_estim(a, t, delta, alpha, lambd, rho, nu, memory, memory_pre))(j)
    #a0 = (delta ** (alpha + 1) / gamma(alpha + 2)) * \
         #((k - 1) ** (alpha + 1) - (k - 1 - alpha) * k ** alpha)
    #aj = (delta ** (alpha + 1) / gamma(alpha + 2)) * \
        #((k - j + 1) ** (alpha + 1) - (k - 1 - j) ** (alpha + 1) - 2 * (k - j) ** (alpha + 1))
    arr_a = np.vectorize(lambda i:a_(i, k, alpha, delta))(j)
    arr_a = np.append(arr_a, a_(k,k, alpha,delta))
    arr_f_h_estim = np.vectorize(lambda i: F(a, h_numerical(a, i, delta, alpha, lambd, rho, nu, memory, memory_pre), lambd, rho, nu))(j)
    hk = np.sum(arr_a[:-1] * arr_f_h_estim + arr_a[-1] * F(a, arr_h_pre_estim, lambd, rho, nu))
    memory[k] = hk
    return hk

def I(r, fk, tk, t, delta):
    return delta/gamma(r)*np.sum((t-tk)**(r-1)*fk)

# TODO : estimate fractional integrals on h
def Lp(theta, v0, a, t, delta, alpha, lambd, rho, nu):
    # approximate the fractional integral with a riemannian sum
    memory = {}
    memory_pre = {}
    n = int (t/delta)
    k = np.arange(1, n + 1)
    hk = np.vectorize(lambda i: h_numerical(a, i, delta, alpha, lambd, rho, nu, memory, memory_pre))(k)
    tk = k*delta
    I1 = I(1, hk, tk, t, delta)
    I2 = I(1-alpha, hk, tk, t, delta)
    return pow(e, theta*lambd*I1 - v0*I2)

theta = 0.04
nu = 0.05
v0 = 0.4
rho = -0.5
lambd = 2
alpha = 1
delta = 0.01
print (Lp(theta, v0, 1, 2, delta, alpha, lambd, rho, nu))
