import numpy as np
import scipy.linalg as linalg


def calculate_normalized_laplacian(adj: np.ndarray) -> np.ndarray:
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj: shape -> [N, N]
    :return:
    """
    d = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    normalized_laplacian = np.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx: np.ndarray) -> np.ndarray:
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = np.diag(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx)
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max = linalg.eigh(L, eigvals_only=True)
        lambda_max = lambda_max[-1]
    M, _ = L.shape
    I = np.eye(M, dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)


def cheb_polynomial(L_tilde, K):
    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials


if __name__ == '__main__':
    adj = np.random.rand(5, 5)
    adj = calculate_scaled_laplacian(adj, lambda_max=None)
    print(adj)
