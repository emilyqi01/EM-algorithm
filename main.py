# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from sklearn.cluster import KMeans


# p(x| mu,cov) = 1/[ (2pi)^(n/2) * |cov|^(1/2) ] * e ^(-1/2 * (X-mu).T * cov.inverse * (X-mu) )
def gaussian(X, mu, cov):
    # show dimensions of X
    # np.linalg.det Compute the determinant of an array.
    n = X.shape[1]
    diff = (X - mu).T
    return np.diagonal(
        1 / ((2 * np.pi) ** (n / 2) * np.linalg.det(cov) ** 0.5) *
        np.exp(-0.5 * np.dot(np.dot(diff.T, np.linalg.inv(cov)), diff))
    ).reshape(-1, 1)


def init_cluster(X, k):
    # k means n clusters
    clusters = []
    index = np.arange(X.shape[0])

    # using k means to initialize Gaussian mixture models

    kmeans = KMeans(k).fit(X)
    mu_k = kmeans.cluster_centers_
    for i in range(k):
        clusters.append(
            {'pi_k': 1.0 / k, 'mu_k': mu_k[i], 'cov_k': np.identity(X.shape[1], dtype=np.float64)}
        )
    return clusters


def E_step(X, clusters):
    global g_nk, totals
    N = X.shape[0]
    K = len(clusters)

    totals = np.zeros((N, 1), dtype=np.float64)
    g_nk = np.zeros((N, K), dtype=np.float64)

    for k, cluster in enumerate(clusters):
        pi_k = cluster['pi_k']
        mu_k = cluster['mu_k']
        cov_k = cluster['cov_k']
        g_nk[:, k] = (pi_k * gaussian(X, mu_k, cov_k)).ravel()

    totals = np.sum(g_nk, 1)
    g_nk = g_nk / np.expand_dims(totals, 1)

def M_step(X,clusters):
    global g_nk
    N = float(X.shape[0])
    # numpy.expand_dims(a, axis)
    # a is array like
    for k, cluster in enumerate(clusters):
        g_k = np.expand_dims(g_nk[:, k], 1)
        N_k = np.sum(g_k, axis=0)
        pi_k = N_k / N
        mu_k = np.sum(g_k * X, axis=0) / N_k
        cov_k = (g_k * (X - mu_k)).T @ (X - mu_k) / N_k

        cluster['pi_k'] = pi_k
        cluster['mu_k'] = mu_k
        cluster['cov_k'] = cov_k

