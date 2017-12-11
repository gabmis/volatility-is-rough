import numpy as np
from math import e, gamma, pi
import matplotlib.pyplot as plt

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
    j = np.arange(k)
    #b = np.vectorize(lambda t: b_(t, k, alpha, delta))(j)
    b = [b_(t, k, alpha, delta) for t in j]
    #h_estim = [h_numerical(a, t, delta, alpha, lambd, rho, nu, memory, memory_pre) for t in j]
    #h_estim = np.vectorize(lambda t: h_numerical(a, t, delta, alpha, lambd, rho, nu, memory, memory_pre))(j)
    h_numerical(a, k-1, delta, alpha, lambd, rho, nu, memory, memory_pre)
    h_estim = [memory[t] for t in j]
    hk_pre = np.sum(b * F(a, np.array(h_estim), lambd, rho, nu))
    memory_pre[k]=hk_pre
    return hk_pre


def h_numerical(a, k, delta, alpha, lambd, rho, nu, memory, memory_pre):
    if k == 0:
        return 0
    if k in memory.keys():
        return memory[k]
    j = np.arange(k)
    #arr_h_pre_estim = [h_pre_estim(a, t, delta, alpha, lambd, rho, nu, memory, memory_pre) for t in j]
    h_pre_estim(a, k, delta, alpha, lambd, rho, nu, memory, memory_pre)
    arr_h_pre_estim = [memory_pre[t] for t in j]
    arr_a = np.vectorize(lambda i:a_(i, k, alpha, delta))(j)
    arr_a = np.append(arr_a, a_(k,k, alpha,delta))
    arr_f_h_estim = np.vectorize(lambda i: F(a, h_numerical(a, i, delta, alpha, lambd, rho, nu, memory, memory_pre), lambd, rho, nu))(j)
    hk = np.sum(arr_a[:-1] * arr_f_h_estim + arr_a[-1] * F(a, np.array(arr_h_pre_estim), lambd, rho, nu))
    memory[k] = hk
    return hk

def I(r, fk, tk, t, delta):
    return delta/gamma(r)*np.sum((t-tk)**(r-1)*fk)

# TODO : estimate fractional integrals on h
def Lp(theta, v0, a, t, delta, alpha, lambd, rho, nu):
    # approximate the fractional integral with a riemannian sum
    memory = {}
    memory[0]=0
    memory_pre = {}
    memory_pre[0]=0
    #print (t, delta)
    n = int (t/delta)
    k = np.arange(n)
    h_numerical(a, n-1, delta, alpha, lambd, rho, nu, memory, memory_pre)
    #hk = [h_numerical(a, i, delta, alpha, lambd, rho, nu, memory, memory_pre) for i in k]
    #hk = np.vectorize(lambda i: h_numerical(a, i, delta, alpha, lambd, rho, nu, memory, memory_pre))(k)
    hk = [memory[idx] for idx in k]
    tk = k*delta
    I1 = I(1, hk, tk, t, delta)
    I2 = I(1-alpha, hk, tk, t, delta)
    return e**(theta*lambd*I1 - v0*I2)

S = 1
K = 0.9
v0 = 0.4
theta = 0.04
v0 = 0.4
delta = 0.01
alpha = 0.9
lambd = 2
rho = -0.5
nu = 0.05
q =0.2
r =0.1
a = 10

#print (Lp(theta, v0, a, 1, delta, alpha, lambd, rho, nu)/a**2)


def payoff(S, K, T, theta, v0, delta, alpha, lambd, rho, nu, q, r):
    k = np.log(S/K) + (r-q)*T
    t1 = S*e**(-q*T)
    t2 = 1/pi*np.sqrt(S*K)*e**(-(r+q)*T/2)
    print (t1, t2)
    u = np.arange(0, 50, delta*10)
    integrande = np.vectorize(lambda i : e**(1j*i*k)*Lp(theta, v0, i-1j/2, T, delta, alpha, lambd, rho, nu)/(i**2+0.25))(u)
    integrande = integrande.real
    return t1-t2*delta*np.sum(integrande)

def plot_payoff(S, K, theta, v0, delta, alpha, lambd, rho, nu, q, r):
    t = np.linspace(0.01, 1, 10)
    pay = np.vectorize(lambda s: payoff(S, K, s, theta, v0, delta, alpha, lambd, rho, nu, q, r))(t)
    plt.plot(t, pay)
    plt.show()

plot_payoff(S, K, theta, v0, delta, alpha, lambd, rho, nu, q, r)
#print(payoff(S, K, 1,theta, v0, delta, alpha, lambd, rho, nu, q, r))
