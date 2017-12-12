import numpy as np
import matplotlib.pyplot as plt
from math import pow
from util import LinkedList
import pandas as pd

mu_ = 0.23
beta = 1.1
Lambd = 1
alpha = 0.6
K1 = 0.3
K2 = (1/2.1253-K1)/beta
phi = []
maxlen = 50
def init(m, T):
    a_T = 1-Lambd/pow(T,alpha)
    mu_T = m/pow(T, 1-alpha)
    N, lambd, dN, mu, phi = np.array([[100.],[0.]]), np.zeros((2,1)), LinkedList(np.zeros((2,1)), maxlen), np.ones((2,1))*mu_T, []
    for t in range (maxlen):
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
    if t<maxlen:
        for s in range(t):
            dn = np.reshape(dnlist[s],(2,1))
            #print (t-s)
            res += np.dot(phi[t-s],dn)
    else:
        for s in range(t-(maxlen-1), t):
            dn = np.reshape(dnlist[s-t+(maxlen-1)],(2,1))
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
    P = [100]
    for t in range(T):
        lambd = mu + integr(t-1, dN, phi)
        dn =evolue(lambd)
        dN.addLast(dn)
        N += dn
        P.append(N[0,0] - N[1,0])
    #print (lambd)
    return P

def figure(T):
    t = np.arange(T)
    m =0
    for i in range (1):
        p = simul(T**2,mu_)
        P = [1/T*p[t*T] for t in range(T)]
        P = P + 2*np.abs(np.min(P))
        #m+= np.mean(P)
        plt.plot(t,P)
    m/=10
    name = "Heston Rough, T = %d, mu = %f, beta = %s, Lambda = %s, alpha = %s, K1 = %s, K2 = %f" %(T, mu_/pow(T,1-alpha),beta, Lambd, alpha, K1, K2)
    #print (m)
    plt.title(name)
    plt.show()
    name += ".csv"
    df = pd.DataFrame(data=P)
    df.to_csv(name, index=False)

figure(5000)

def save(T):
    p = simul(T**2,mu_)
    P = [np.sqrt(Lambd/(pow(T,alpha)*mu_*T**alpha))*p[t*T] for t in range(T)]
    df = pd.DataFrame(data=P)
    name = "Heston Rough, h = %d, mu = %f, beta = %s, Lambda = %s, alpha = %s, K1 = %s, K2 = %f" %(T, mu_/pow(T,1-alpha),beta, Lambd, alpha, K1, K2)
    name+=".csv"
    df.to_csv(name, index=False)

#save(10)

def ray():
    t = np.linspace(0,1000, 10001)
    dt = 1./10
    s = np.zeros((2,2))
    for i in t:
        s += i*PHI(i*dt)
    s*=dt**2
    return np.max(np.abs(np.linalg.eig(s)[0]))

#print (ray())
