import numpy as np
import matplotlib.pyplot as plt
from math import e, pow

mu_ = 0.1
beta = 1.05
gamma = 1
alpha = 1 + beta/gamma
phi = []

def init(m, T):
    N, lambd, dN, mu, phi = np.zeros((2,1)), np.zeros((2,1)), np.zeros((2,1)), np.ones((2,1))*m, []
    for t in range (T):
        phi.append(PHI(t))
    return N, lambd, dN, mu, phi

def phi_1(t):
    return pow(e,-alpha*t)

def phi_2(t):
    return pow(e, -gamma*t)

def PHI(t):
    return np.array([[phi_1(t), beta*phi_2(t)],[phi_2(t), phi_1(t) + (beta-1)*phi_2(t)]])

def integr(t, dN, phi):
    res = np.zeros((2,1))
    for s in range(t):
        dn = np.reshape(dN[:,s],(2,1))
        res += np.dot(phi[t-s],dn)
    return res

def evolue(x, lambd):
    dn = np.zeros((2,1))
    if (x[0] <= lambd[0,0]):
        dn[0,0] = 1
    if (x[1] <= lambd[1,0]):
        dn[1,0] = 1
    return dn

def simul(T,m):
    N, lambd, dN, mu, phi= init(m, T)
    P = [0]
    for t in range(T):
        lambd = mu + integr(t-1, dN, phi)
        x = np.random.random(2)
        dn =evolue(x, lambd)
        dN = np.append(dN, dn, axis=1)
        N += dn
        P.append(N[0,0] - N[1,0])
    #print (lambd)
    return P

def figure(T):
    t = np.arange(T+1)
    for i in range (15):
        P = simul(T,mu_)
        plt.plot(t,P)
    plt.show()

figure(10000)
