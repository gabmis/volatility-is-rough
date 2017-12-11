import numpy as np

## parameters
# time step
delta = 1
# hawkes parameters
xsi = 0.2
alpha = 0.6
theta = 0.2
beta = 2
c = 1 # limit of a specific function (page 6) related to the base hawkes process sequence
lambd_star = 1 # limit of a specific function (page 6) related to the base hawkes process sequence
lambd = lambd_star * alpha / c * np.euler_gamma(1 - alpha)
mu = 1 # >0 depends on Hawkes process
# rough heston parameters
v0 = xsi * theta
rho = (1 - beta) / np.sqrt(2 * (1 + beta ** 2))
nu = np.sqrt((theta * (1 + beta ** 2)) / (lambd * mu * (1 + beta) ** 2))