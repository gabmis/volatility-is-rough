import numpy as np
import matplotlib.pyplot as plt
from math import pow
from util import LinkedList2

mu_ = 0.1
beta = 1.1
Lambd = 1
alpha = 0.6
K1 = 0.3
K2 = (1/2.1253-K1)/beta
phi = []

def init(m, T):
    a_T = 1-Lambd/pow(T,alpha)
    mu_T = m/pow(T, 1-alpha)
    N, lambd, dN, mu, phi = np.zeros((2,1)), np.zeros((2,1)), LinkedList2(np.zeros((2,1))), np.ones((2,1))*mu_T, []
    for t in range (630):
        phi.append(a_T*PHI(t))
    return N, lambd, dN, mu, phi

def phi_1(t):
    return K1/(1+pow(t,1+alpha))

def phi_2(t):
    return K2/(1+pow(t,1+alpha))

def PHI(t):
    return np.array([[phi_1(t), beta*phi_2(t)],[phi_2(t), phi_1(t) + (beta-1)*phi_2(t)]])

def integr(t, dN, phi):
    res = np.zeros((2,1))
    dnlist = dN.getValues()
    if t<630:
        for s in range(t):
            dn = np.reshape(dnlist[s],(2,1))
            #print (t-s)
            res += np.dot(phi[t-s],dn)
    else:
        for s in range(t-629, t):
            dn = np.reshape(dnlist[s-t+629],(2,1))
            res += np.dot(phi[t-s],dn)
    return res

def evolue(lambd):
    x = np.random.random(2)
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
        dn =evolue(lambd)
        dN.addLast(dn)
        N += dn
        P.append(N[0,0] - N[1,0])
    #print (lambd)
    return P

def figure(T):
    t = np.arange(T+1)
    m =0
    for i in range (1):
        P = simul(T,mu_)
        #m+= np.mean(P)
        plt.plot(t,P)
    m/=10
    #print (m)
    plt.show()

figure(100000)
