import numpy as np
import numpy.random as rd
import scipy.linalg as lalg


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
	coefs[0] = 0
	coefs[1:n + 1] = vec_cov(h, index[:n])
	coefs[n + 1:] = vec_cov(h, m - index[n:])
	return coefs

# circulant matrix


def mat_circ(h, n):
	coefs = circ_coefs(h, n)
	return lalg.circulant(coefs)

# fft matrix and diagonal matrix with sqrt of eigenvalues
# (COULD BE DONE USING FFT LIBRARY FROM NUMPY ---> FASTER?)


def q_and_diag_sqrt(h, n):
	m = 2 * (n - 1)
	j = np.zeros((m, m))
	j[1:, :] = np.ones((m - 1, m))
	j = np.cumsum(j, axis=0)
	k = np.zeros((m, m))
	k[:, 1:] = np.ones((m, m - 1))
	k = np.cumsum(k, axis=1)
	q = 1 / np.sqrt(m) * np.exp(-2 * 1j * np.pi * (1 / m) * j * k)
	coefs = circ_coefs(h, n)
	q_t = np.transpose(q)
	eigen_complex = np.dot(q_t, coefs)
	eigen = np.real(eigen_complex)
	eigen[eigen < 0] = 0  # GET RID OF NEGATIVE VALUES TODO: FIX
	eigen_sqrt = np.sqrt(eigen)
	diag_sqrt = np.diag(eigen_sqrt)
	return q, diag_sqrt


def fbm(h, n):
	m = 2 * (n - 1)
	x = rd.normal(m)
	q, diag_sqrt = q_and_diag_sqrt(h, n)
	q_adj = np.conj(np.transpose(q))
	s = np.dot(np.dot(q, diag_sqrt), q_adj)
	fgn = np.real(np.dot(s, x)[:n])
	fbm = np.cumsum(fgn * (1 / n) ** h)
	return fbm