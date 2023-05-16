import numpy as np
import scipy as sp
import scipy.linalg
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt


## least-squares solved via single SVD
def SVD(M, r):  # input matrix M, approximating with rank r
    u, s, vh = np.linalg.svd(M, full_matrices=False)  # s is diag
    X = u[:, :r].dot(np.diag(np.sqrt(s[:r])))
    Y = vh[:r, :].T.dot(np.diag(np.sqrt(s[:r])))
    return X.dot(Y.T)  # , X, Y ## least-squares solved via single SVD


## hard_impute for matrix completion
def hard_impute(O, Ω, r=1, eps=1e-4):
    M = np.zeros_like(O)
    for T in range(2000):
        M_new = SVD(O * Ω + (1 - Ω) * M, r)
        if np.linalg.norm(M - M_new) < np.linalg.norm(M) * eps:
            break
        M = M_new
    return M


def noise_to_signal(X, M, Ω):
    return np.sqrt(np.sum((Ω * X - Ω * M) ** 2) / np.sum((Ω * M) ** 2))


def abs_mean(X, M, Ω):
    return np.sum(np.abs((X - M) * Ω)) / np.sum(Ω)


def low_rank_matrix_generation(n1, n2, r, gamma_shape, gamma_scale, mean_M):
    """
    generate a low-rank matrix M0
    low rank components are entry-wise Gamma distribution
    mean_M: the mean value of M0
    """
    U = np.random.gamma(shape=gamma_shape, scale=gamma_scale, size=(n1, r))
    V = np.random.gamma(shape=gamma_shape, scale=gamma_scale, size=(n2, r))
    M0 = U.dot(V.T)
    M0 = M0 / np.mean(M0) * mean_M
    return M0


def compute_Sigma_Poisson(M0, r, p_observe):
    """
    Compute the standard deviation of least-square estimator for Poisson noise

    The formula is specified in the paper Eq.(10): https://arxiv.org/pdf/2110.12046.pdf
    """
    u, s, vh = np.linalg.svd(M0, full_matrices=False)
    U = u[:, :r]
    V = vh[:r, :].T

    sigmaS = ((U.dot(U.T)) ** 2).dot(M0) + M0.dot((V.dot(V.T)) ** 2)
    sigmaS /= p_observe
    sigmaS = np.sqrt(sigmaS)

    return sigmaS


def compute_Sigma_Bernoulli(M0, r, p_observe):
    u, s, vh = np.linalg.svd(M0, full_matrices=False)
    U = u[:, :r]
    V = vh[:r, :].T

    sigmaS = ((U.dot(U.T)) ** 2).dot(M0 * (1 - M0)) + (M0 * (1 - M0)).dot(
        (V.dot(V.T)) ** 2
    )
    sigmaS /= p_observe
    sigmaS = np.sqrt(sigmaS)

    return sigmaS


def compute_Sigma_adaptive(Mhat, E, r, p_observe):
    """
    Compute the standard deviation of least-square estimator for arbitrary noise

    The formula is specified in the paper Corollary 1: https://arxiv.org/pdf/2110.12046.pdf
    """
    u, s, vh = np.linalg.svd(Mhat, full_matrices=False)
    U = u[:, :r]
    V = vh[:r, :].T

    sigmaS = ((U.dot(U.T)) ** 2).dot(E**2) + (E**2).dot((V.dot(V.T)) ** 2)

    sigmaS /= p_observe**2
    sigmaS = np.sqrt(sigmaS)

    return sigmaS
