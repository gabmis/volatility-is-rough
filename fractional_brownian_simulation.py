import numpy as np
import numpy.random as rd
import scipy.linalg as lalg
from scipy.fftpack import fft, ifft


# our covariance function
def cov(h, n):
    if n >= 1:
        return 0.5 * ((n + 1) ** (2 * h) + (n - 1) ** (2 * h)
                      - 2 * n ** (2 * h))
    else:
        print("index 0 in cov function")
        return False

# vectorize it for further use
vec_cov = np.vectorize(cov)


# circulant coefficients function:
def circ_coefs(h, n):
    m = 2 * (n - 1)
    index = np.arange(1, m)
    coefs = np.zeros(m)
    coefs[0] = 1
    coefs[1:n] = vec_cov(h, index[:n - 1])
    coefs[n:] = vec_cov(h, m - index[n - 1:])
    return coefs


# circulant matrix
def mat_circ(h, n):
    coefs = circ_coefs(h, n)
    return lalg.circulant(coefs)


# fft matrix and diagonal matrix with sqrt of eigenvalues
def eigen_values(h, n):
    m = 2 * (n - 1)
    j = np.ones((m, m))
    j[0, :] = np.zeros(m)
    j = np.cumsum(j, axis=0)
    k = np.zeros((m, m))
    k[:, 1:] = np.ones((m, m - 1))
    k = np.cumsum(k, axis=1)
    q = 1 / np.sqrt(m) * np.exp(-2 * 1j * np.pi * (1 / m) * j * k)
    coefs = circ_coefs(h, n)
    q_t = np.transpose(q)
    eigen_complex = np.sqrt(m) * np.dot(q_t, coefs)
    eigen_sqrt = np.sqrt(eigen_complex)
    return eigen_sqrt


def fbm(h, n, t):
    m = 2 * (n - 1)
    # random gaussian
    seed = 10^2
    rd.seed(seed)
    xsi = rd.normal(size=m)
    # computing fbm
    # multiply by 1/sqrt(m)Q* by taking inverse dft
    step1 = ifft(xsi)
    # multiply by vector of eigenvalues
    step2 = np.multiply(step1, eigen_values(h, n))
    # multiply by sqrt(m)Q by taking dft
    fgn = np.real(fft(step2))[:n]
    res = (t / n) ** h * fgn.cumsum()
    return res

