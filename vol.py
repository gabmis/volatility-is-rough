import numpy as np
import scipy.stats
from math import e
import matplotlib.pyplot as plt

S = 1
K = 1
r=0.05
T = 2
C0 = 1.01

def f(C0, S, K, r, T, sigma):
    d1 = (np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = (np.log(S/K)+(r-0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    return C0 - S*scipy.stats.norm.cdf(d1) - K*e**(-r*T)*scipy.stats.norm.cdf(d2)



t = np.linspace(0.01, 10, 100)
z = np.zeros(t.shape)

for i in range(t.shape[0]):
    z[i] = f(C0, S, K, r, T, t[i])
plt.plot(t,z)
plt.show()
