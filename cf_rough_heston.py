import numpy as np
from math import e, gamma, pi
import matplotlib.pyplot as plt

# k here is k+1 in the paper
def F(a, x, lambd, rho, nu):
    return -a/2 * (a  + 1j) + lambd * (1j * a * rho * nu - 1) * x + (lambd * nu * x) ** 2/2

def a_(j,k, alpha, delta):
    if j == 0:
        return delta**alpha/gamma(alpha+2)*((k-1)**alpha+1-(k-1-alpha)*k**alpha)
    elif j<k:
        return delta**alpha/gamma(alpha+2)*((k+1-j)**(alpha+1) + (k-1-j)**(alpha+1)-2*(k-j)**(alpha+1))
    else :
        return delta**alpha/gamma(alpha+2)

def b_(j, k, alpha, delta):
    return delta**alpha/gamma(alpha+1)*((k-j)**alpha-(k-1-j)**alpha)

def h_pre_estim(a, k, delta, alpha, lambd, rho, nu, memory, memory_pre):
    if k in memory_pre.keys():
        return memory_pre[k]
    j = np.arange(k)
    #b = np.vectorize(lambda t: b_(t, k, alpha, delta))(j)
    b = [b_(i, k, alpha, delta) for i in j]
    #h_estim = [h_numerical(a, t, delta, alpha, lambd, rho, nu, memory, memory_pre) for t in j]
    #h_estim = np.vectorize(lambda t: h_numerical(a, t, delta, alpha, lambd, rho, nu, memory, memory_pre))(j)
    h_numerical(a, k-1, delta, alpha, lambd, rho, nu, memory, memory_pre)
    h_estim = [memory[i] for i in j]
    hk_pre = np.sum(b * F(a, np.array(h_estim), lambd, rho, nu))
    memory_pre[k]=hk_pre
    return hk_pre


def h_numerical(a, k, delta, alpha, lambd, rho, nu, memory, memory_pre):
    if k in memory.keys():
        return memory[k]
    j = np.arange(k)
    #arr_h_pre_estim = [h_pre_estim(a, t, delta, alpha, lambd, rho, nu, memory, memory_pre) for t in j]
    ajk = np.vectorize(lambda i:a_(i, k, alpha, delta))(j)
    akk = a_(k,k, alpha,delta)
    h_pre_k = h_pre_estim(a, k, delta, alpha, lambd, rho, nu, memory, memory_pre)
    arr_f_h_estim = np.vectorize(lambda i: F(a, h_numerical(a, i, delta, alpha, lambd, rho, nu, memory, memory_pre), lambd, rho, nu))(j)
    hk = np.sum(ajk * arr_f_h_estim) + akk * F(a, h_pre_k, lambd, rho, nu)
    memory[k] = hk
    return hk

def I(r, hk, tk, t, delta):
    return delta/gamma(r)*np.sum((t-tk)**(r-1)*hk)

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
    tk = k*delta
    h_numerical(a, n, delta, alpha, lambd, rho, nu, memory, memory_pre)
    #hk = [h_numerical(a, i, delta, alpha, lambd, rho, nu, memory, memory_pre) for i in k]
    #hk = np.vectorize(lambda i: h_numerical(a, i, delta, alpha, lambd, rho, nu, memory, memory_pre))(k)
    hk = [memory[idx] for idx in k]
    I1 = I(1, hk, tk, t, delta)
    I2 = I(1-alpha, hk, tk, t, delta)
    return e**(theta*lambd*I1 - v0*I2)

S = 1
K = 1
theta = 0.04
v0 = 0.4
delta = 0.01
alpha = 0.6
lambd = 2
rho = -0.5
nu = 0.05
q =0
r =0.05
a = 10

#print (Lp(theta, v0, a-1j/2, 1, delta, alpha, lambd, rho, nu))

def payoff(S, K, T, theta, v0, delta, alpha, lambd, rho, nu, q, r):
    k = np.log(S/K) + (r-q)*T
    t1 = S*e**(-q*T)
    t2 = 1/pi*np.sqrt(S*K)*e**(-(r+q)*T/2)
    D = .1
    u = np.linspace(0, 5, 50)
    integrande = np.vectorize(lambda i : e**(1j*i*k)*Lp(theta, v0, i-1j/2, T, delta, alpha, lambd, rho, nu)/(i**2+0.25))(u)
    integrande = integrande.real
    return t1-t2*D*np.sum(integrande)

def plot_payoff(S, K, T, theta, v0, delta, alpha, lambd, rho, nu, q, r):
    k = np.linspace(0.01, 2, 20)
    pay = np.vectorize(lambda s: payoff(S, s, T, theta, v0, delta, alpha, lambd, rho, nu, q, r))(k)
    plt.plot(k, pay)
    plt.xlabel('maturité')
    plt.ylabel('payoff')
    plt.show()

#plot_payoff(S, K, 2, theta, v0, delta, alpha, lambd, rho, nu, q, r)
#print(payoff(S, K, 1, theta, v0, delta, alpha, lambd, rho, nu, q, r))

from vol import f

def sigma(S, K, T, theta, v0, delta, alpha, lambd, rho, nu, q, r):
    #C0 = payoff(S, K, T, theta, v0, delta, alpha, lambd, rho, nu, q, r)
    C0 = 1.1
    i = 0.01
    print(f(C0, S, K, r, T, i))
    if (f(C0, S, K, r, T, i)>0):
        t = np.linspace(0.0001, i, 100)
    else :
        while (f(C0, S, K, r, T, i)<0):
            i+=1
        t = np.linspace(i-1, i, 100)
    j=0
    while (f(C0, S, K, r, T, t[j])<0):
        j+=1
    return (t[j-1]+t[j])/2

print (sigma(S, K, 2, theta, v0, delta, alpha, lambd, rho, nu, q, r))
