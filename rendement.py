import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Heston, h = 1000000, mu = 0.526, beta = 1.07, Lambda = 1, K1 = 2.150538, K2 = 2, a_T = 0.999999.csv")
p = df.values
T = p.shape[0]
P = np.reshape(p, (T,))
P = P + 2*np.abs(np.min(P))
rend = np.zeros(10000)

for i in range(10000):
    rend[i] = np.log(P[i+1]/P[i])

plt.plot(rend)
plt.title("log rendement, motion Heston, h=2e3")
plt.show()
