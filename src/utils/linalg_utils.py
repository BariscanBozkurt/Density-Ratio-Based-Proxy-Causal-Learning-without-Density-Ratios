from jax import jit
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from typing import Tuple


@jit
def pairwise_squared_distance(X: jnp.ndarray, Y: jnp.ndarray = None) -> jnp.ndarray:
    """
    Returns the pairwised squared distances

    Parameters
    -----------
    X: jax ndarray of shape (nsamples, n_features)
    Y: jax ndarray of shape (nsamples, n_features), optional
        If not provided, defaults to X.

    Returns
    -------
    jax ndarray
        Pairwise squared distances of shape (nsamples, nsamples).
    """
    if Y is None:
        Y = X
        
    return ((X[:, None] - Y[None, :]) ** 2).sum(-1)


@jit
def pairwise_absolute_distance(X: jnp.ndarray, Y: jnp.ndarray = None) -> jnp.ndarray:
    """
    Returns the pairwised squared distances

    Parameters
    -----------
    X: jax ndarray of shape (nsamples, n_features)
    Y: jax ndarray of shape (nsamples, n_features), optional
        If not provided, defaults to X.

    Returns
    -------
    jax ndarray
        Pairwise squared distances of shape (nsamples, nsamples).
    """
    if Y is None:
        Y = X
        
    return jnp.abs(X[:, None] - Y[None, :]).sum(-1)


def make_psd(A: jnp.ndarray, eps: float = 1e-10) -> jnp.ndarray:
    """
    Ensure the matrix is positive semi-definite (PSD).

    Given a matrix A, this function ensures it is positive semi-definite by making it symmetric
    and adding a small positive value to its diagonal.

    Parameters:
    - A (np.ndarray): Input matrix.

    Returns:
    - np.ndarray: Positive semi-definite matrix.
    """
    n = A.shape[0]
    return (A + A.T) / 2 + eps * jnp.eye(n)


def cartesian_product(*arrays: np.ndarray) -> np.ndarray:
    """
    Compute the Cartesian product of input arrays.

    Args:
        *arrays: Variable number of input arrays.

    Returns:
        np.ndarray: Cartesian product of the input arrays.

    Raises:
        ValueError: If the input arrays are empty.

    Example:
        >>> stage1_idx = np.array([1, 2, 3])
        >>> stage2_idx = np.array([4, 5])
        >>> cart = cartesian_product(stage1_idx, stage2_idx)
        >>> print(cart)
        [[1 4]
         [1 5]
         [2 4]
         [2 5]
         [3 4]
         [3 5]]
    """
    if not arrays:
        raise ValueError("No arrays provided for Cartesian product.")
    
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


def remove_diagonal_elements(A):
    return A - jnp.diag(jnp.diag(A))

def columns_mean_excluding_self(A):
    B = remove_diagonal_elements(jnp.ones((A.shape[1], A.shape[1]))) / (A.shape[1] - 1)
    return A @ B

def IncompleteCholesky(K: jnp.ndarray, eta: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Incomplete Cholesky Decomposition.

    Given a positive definite kernel matrix K and a threshold eta, this function performs an incomplete
    Cholesky decomposition as described on Page 129 of "Kernel Methods and Pattern Analysis" by John Taylor
    and Mello Cristianimi.

    The output consists of a matrix L such that L.T @ L = K. Note that L is not necessarily triangular.
    Additionally, the function returns a permutation matrix P such that P.T @ L.T @ L @ P = P.T @ K @ P,
    where L @ P is upper triangular.

    Parameters:
    - K (jnp.ndarray): Positive definite kernel matrix.
    - eta (float): Threshold for stopping the decomposition.

    Returns:
    - tuple[jnp.ndarray, jnp.ndarray]: Tuple containing:
        - L (jnp.ndarray): Matrix such that L.T @ L = K.
        - P (jnp.ndarray): Permutation matrix such that P.T @ L.T @ L @ P = P.T @ K @ P, where L @ P is upper triangular.
    """
    j = 0
    ell = K.shape[0]
    L = jnp.zeros((ell, ell))
    d = jnp.diag(K).copy()
    d_max_idx_list = []

    d_max_idx = jnp.argmax(d)
    d_max_idx_list.append(d_max_idx)
    d_max = d[d_max_idx]

    while d_max > eta:
        nu = jnp.sqrt(d_max)
        for i in tqdm(range(ell)):
            L = L.at[j, i].set((K[d_max_idx_list[j], i] - jnp.dot(L[:, i], L[:, d_max_idx_list[j]])) / nu)
            d = d.at[i].set(d[i] - L[j, i] ** 2)

        j += 1
        d_max_idx = jnp.argmax(d)
        d_max_idx_list.append(d_max_idx)
        d_max = d[d_max_idx]
        print(d_max)
    L = L[:j, :]

    # Construct permutation matrix P based on the absolute values of L exceeding a small threshold
    P = jnp.eye(ell)[jnp.argsort((jnp.abs(L) > 1e-7).sum(0))]
    
    return L, P


def IncompleteCholesky_np(K: np.ndarray, eta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Incomplete Cholesky Decomposition.

    Given a positive definite kernel matrix K and a threshold eta, this function performs an incomplete
    Cholesky decomposition as described on Page 129 of "Kernel Methods and Pattern Analysis" by John Taylor
    and Mello Cristianimi.

    The output consists of a matrix L such that L.T @ L = K. Note that L is not necessarily triangular.
    Additionally, the function returns a permutation matrix P such that P.T @ L.T @ L @ P = P.T @ K @ P,
    where L @ P is upper triangular.

    Parameters:
    - K (np.ndarray): Positive definite kernel matrix.
    - eta (float): Threshold for stopping the decomposition.

    Returns:
    - tuple[np.ndarray, np.ndarray]: Tuple containing:
        - L (np.ndarray): Matrix such that L.T @ L = K.
        - P (np.ndarray): Permutation matrix such that P.T @ L.T @ L @ P = P.T @ K @ P, where L @ P is upper triangular.
    """
    j = 0
    ell = K.shape[0]
    L = np.zeros((ell, ell))
    d = np.diag(K).copy()
    d_max_idx_list = []

    d_max_idx = np.argmax(d)
    d_max_idx_list.append(d_max_idx)
    d_max = d[d_max_idx]

    while d_max > eta:
        nu = np.sqrt(d_max)
        for i in tqdm(range(ell)):
            L[j, i] = (K[d_max_idx_list[j], i] - L[:, i].T @ L[:, d_max_idx_list[j]]) / nu
            d[i] = d[i] - L[j, i] ** 2

        j += 1
        d_max_idx = np.argmax(d)
        d_max_idx_list.append(d_max_idx)
        d_max = d[d_max_idx]
        print(d_max)
    L = L[:j, :]

    # Construct permutation matrix P based on the absolute values of L exceeding a small threshold
    P = np.eye(ell)[np.argsort((np.abs(L) > 1e-7).sum(0))]
    
    return L, P
