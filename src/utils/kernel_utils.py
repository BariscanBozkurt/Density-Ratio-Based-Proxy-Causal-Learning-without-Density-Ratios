import numpy as np
from jax import jit
import jax.numpy as jnp
from scipy.special import loggamma
import sympy
from typing import Optional, Union, Tuple, Dict, List, Any
from abc import ABC, abstractmethod
from functools import partial

from .linalg_utils import pairwise_squared_distance, pairwise_absolute_distance

from jax import config
config.update("jax_enable_x64", True)

def GaussianKernelMatrix(data1: jnp.ndarray, data2: jnp.ndarray = None, bandwidth: float = 0.5) -> jnp.ndarray:
    """
    Compute the Gaussian Kernel Matrix.

    Given a dataset and an optional Gaussian kernel parameter, this function computes the Gaussian
    kernel matrix.

    Parameters:
    - data1, data2 (jnp.ndarray): Input data matrices with shape (n_samples, n_features).
    - bandwidth (float, optional): Standard deviation parameter for the Gaussian kernel. Default is 0.5.

    Returns:
    - jnp.ndarray: Gaussian kernel matrix with shape (n_samples, n_samples).
    """
    if data2 is None:
        data2 = data1
    # Ensure data is a NumPy array
    data1 = jnp.array(data1, dtype = jnp.float64)
    data2 = jnp.array(data2, dtype = jnp.float64)
    # Calculate pairwise Euclidean distances
    # pairwise_dists = squareform(cdist(data1, data2, "euclidean"))
    pairwise_dists = pairwise_squared_distance(data1, data2)
    # Compute Gaussian kernel matrix
    kernel = jnp.exp((-(pairwise_dists)) / (2 * bandwidth ** 2))
    
    return kernel


class Kernel(ABC):
    """
    Base class for all kernels.
    """
    @abstractmethod
    def __call__(self, 
                 X: jnp.ndarray, 
                 Y: Optional[jnp.ndarray] = None,
                 ) -> jnp.ndarray:
        ...

    def __repr__(self) -> str:
        ...

    def get_params(self):
        ...

    def set_params(self, **params):
        ...


class DifferentiableKernel(Kernel, ABC):
    """
    Base class for differentiable kernels.
    """
    @abstractmethod
    def tree_flatten(self) -> Tuple[Tuple, Dict[str, Any]]:
        # Taken from https://github.com/jswu18/distribution-discrepancies/blob/main/kernels.py
        """
        To have JIT-compiled class methods by registering the type as a custom PyTree object.
        As referenced in:
        https://jax.readthedocs.io/en/latest/faq.html#strategy-3-making-customclass-a-pytree

        :return: A tuple containing dynamic and a dictionary containing static values
        """
        raise NotImplementedError("Needs to implement tree_flatten")


class StackOfKernels(Kernel):

    def __init__(self, list_of_kernels, **kwargs):
        self.list_of_kernels = list_of_kernels

    def __call__(self, 
                 X: jnp.ndarray, 
                 Y: Optional[jnp.ndarray] = None,
                 ) -> jnp.ndarray:
        if Y is None:
            Y = X
        data_shape = X.shape[0]
        K = jnp.ones((data_shape, data_shape))
        for jj in range(len(self.list_of_kernels)):
            K *= self.list_of_kernels[jj](X[:, jj].reshape(-1, 1), Y[:, jj].reshape(-1, 1))

        return K
    

class BinaryKernel(Kernel):

    def __init__(self, **kwargs):
        # super(BinaryKernel, self).__init__()
        pass

    def __call__(self, X, Y = None):
        if Y is None:
            Y = X
        res = X.dot(Y.T)
        res += (1 - X).dot(1 - Y.T)
        return res


class LinearKernel(Kernel):
    
    def __init__(self, **kwargs) -> None:
        pass

    def __call__(self, 
                 X: jnp.ndarray, 
                 Y: Optional[jnp.ndarray] = None,
                 ) -> jnp.ndarray:
        if Y is None:
            Y = X
        return X.dot(Y.T)
    
    def __repr__(self) -> str:
        """
        Return a string representation of the Linear kernel.

        Returns
        -------
        str
            String representation of the Linear kernel.
        """
        raise NotImplementedError

    def get_params(self) -> Dict[str, float]:
        """
        Get the parameters of the kernel.

        Returns
        -------
        Dict[str, float]
            Dictionary containing the parameters of the kernel.
        """
        raise NotImplementedError

    def set_params(self, **params):
        """
        Set the parameters of the kernel.
        """
        raise NotImplementedError
    

class RBF(Kernel):
    """
    Radial Basis Function (RBF) kernel.

    Parameters
    ----------
    length_scale : float, optional
        The length scale parameter of the RBF kernel.

    Attributes
    ----------
    length_scale : float
        The length scale parameter of the RBF kernel.
    """
    def __init__(self, 
                 length_scale: float = 0.5,
                 use_length_scale_heuristic: bool = False,
                 length_scale_heuristic_quantile: float = 0.5,
                 use_jit_call: bool = False) -> None:
        self.length_scale = length_scale
        self.use_length_scale_heuristic = use_length_scale_heuristic
        self.length_scale_heuristic_quantile = length_scale_heuristic_quantile
        self.use_jit_call = use_jit_call    
        

    def __call__(self, 
                 X: jnp.ndarray, 
                 Y: Optional[jnp.ndarray] = None,
                 ) -> jnp.ndarray:
        """
        Return the RBF kernel k(X, Y) and (optionally its gradient will be added later in the future)

        Parameters
        ----------
        X : jnp.ndarray
            Left argument of the returned kernel k(X, Y).
        Y : jnp.ndarray, optional
            Right argument of the returned kernel k(X, Y). If None, k(X, X) is evaluated instead.

        Returns
        -------
        jnp.ndarray
            Kernel k(X, Y)
        """
        if Y is None:
            Y = X

        if self.use_jit_call:
            K, length_scale = self.call(X, Y, self.length_scale, self.use_length_scale_heuristic, self.length_scale_heuristic_quantile)
            self.length_scale = float(length_scale)
            return K
        else:
            distances = pairwise_squared_distance(X, Y)
            if self.use_length_scale_heuristic:
                # length_scale_squared = jnp.median(distances[jnp.triu_indices(distances.shape[0], k = 1)]) / 2.0
                length_scale_squared = jnp.quantile(distances[jnp.triu_indices(distances.shape[0], k = 1)], self.length_scale_heuristic_quantile) / 2.0
                K = jnp.exp( - distances / (2 * length_scale_squared))
                self.length_scale = float(jnp.sqrt(length_scale_squared))
            else:
                K = jnp.exp( - distances / (2 * self.length_scale ** 2))
            return K
        
    @staticmethod
    @partial(jit, static_argnums=3) 
    def call(X: jnp.ndarray, 
             Y: Optional[jnp.ndarray] = None,
             length_scale: float = 1.0,
             use_length_scale_heuristic: Optional[bool] = False,
             length_scale_heuristic_quantile: float = 0.5) -> jnp.ndarray:
        """
        Compute the RBF kernel k(X, Y).

        Parameters
        ----------
        X : jnp.ndarray
            Left argument of the kernel.
        Y : jnp.ndarray, optional
            Right argument of the kernel. If None, k(X, X) is evaluated.
        length_scale : float, optional
            The length scale parameter of the RBF kernel.

        Returns
        -------
        jnp.ndarray
            Computed RBF kernel k(X, Y)
        """
        if Y is None:
            Y = X
        distances = pairwise_squared_distance(X, Y)
        if use_length_scale_heuristic:
            # length_scale_squared = jnp.median(distances[jnp.triu_indices(distances.shape[0], k = 1)]) / 2
            length_scale_squared = jnp.quantile(distances[jnp.triu_indices(distances.shape[0], k = 1)], length_scale_heuristic_quantile) / 2.0
            K = jnp.exp( - distances / (2 * length_scale_squared))
            return K, jnp.sqrt(length_scale_squared)
        else:
            K = jnp.exp( - distances / (2 * length_scale ** 2))
            return K, length_scale
        
    def __repr__(self) -> str:
        """
        Return a string representation of the RBF kernel.

        Returns
        -------
        str
            String representation of the RBF kernel.
        """
        raise NotImplementedError

    def get_params(self) -> Dict[str, float]:
        """
        Get the parameters of the kernel.

        Returns
        -------
        Dict[str, float]
            Dictionary containing the parameters of the kernel.
        """
        return {'length_scale': self.length_scale}

    def set_params(self, **params):
        """
        Set the parameters of the kernel.
        """
        for param_name, param_value in params.items():
            setattr(self, param_name, param_value)

    def tree_flatten(self) -> Tuple[Tuple, Dict[str, Any]]:
        children = ()
        aux_data = {"length_scale": self.length_scale}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls, aux_data: Dict[str, Any], children: Tuple
    ):
        return cls(*children, **aux_data)


class SquaredExponential(Kernel):
    """
    Squared Exponential kernel.

    Parameters
    ----------
    theta : float
        amplitude of the kernel

    length_scale : float, optional
        The length scale parameter of the RBF kernel.

    Attributes
    ----------
    theta : float
        amplitude of the kernel

    length_scale : float
        The length scale parameter of the RBF kernel.
    """
    def __init__(self, 
                 theta: float = 1,
                 length_scale: float = 0.5,
                 use_length_scale_heuristic: bool = False,
                 use_jit_call: bool = False) -> None:
        
        self.theta = theta
        self.length_scale = length_scale
        self.use_length_scale_heuristic = use_length_scale_heuristic
        self.use_jit_call = use_jit_call    
        

    def __call__(self, 
                 X: jnp.ndarray, 
                 Y: Optional[jnp.ndarray] = None,
                 ) -> jnp.ndarray:
        """
        Return the Squared Exponential kernel k(X, Y) and (optionally its gradient will be added later in the future)

        Parameters
        ----------
        X : jnp.ndarray
            Left argument of the returned kernel k(X, Y).
        Y : jnp.ndarray, optional
            Right argument of the returned kernel k(X, Y). If None, k(X, X) is evaluated instead.

        Returns
        -------
        jnp.ndarray
            Kernel k(X, Y)
        """
        if Y is None:
            Y = X

        if self.use_jit_call:
            K, length_scale = self.call(X, Y, self.theta, self.length_scale, self.use_length_scale_heuristic)
            self.length_scale = length_scale
            return K
        else:
            distances = pairwise_squared_distance(X, Y)
            if self.use_length_scale_heuristic:
                length_scale_squared = jnp.median(distances[jnp.triu_indices(distances.shape[0], k = 1)]) / 2
                K = (self.theta ** 2) * jnp.exp( - distances / (2 * length_scale_squared))
                self.length_scale = jnp.sqrt(length_scale_squared)
            else:
                K = (self.theta ** 2) * jnp.exp( - distances / (2 * self.length_scale ** 2))
            return K
        
    @staticmethod
    @partial(jit, static_argnums=(4,)) 
    def call(X: jnp.ndarray, 
             Y: Optional[jnp.ndarray] = None,
             theta: float = 1.0, 
             length_scale: float = 1.0,
             use_length_scale_heuristic: Optional[bool] = False) -> jnp.ndarray:
        """
        Compute the Squared Exponential kernel k(X, Y).

        Parameters
        ----------
        X : jnp.ndarray
            Left argument of the kernel.
        Y : jnp.ndarray, optional
            Right argument of the kernel. If None, k(X, X) is evaluated.
        theta: float, optional
            Amplitude of the Squared Exponential kernel
        length_scale : float, optional
            The length scale parameter of the Squared Exponential kernel.

        Returns
        -------
        jnp.ndarray
            Computed Squared Exponential kernel k(X, Y)
        """
        if Y is None:
            Y = X
        distances = pairwise_squared_distance(X, Y)
        if use_length_scale_heuristic:
            length_scale_squared = jnp.median(distances[jnp.triu_indices(distances.shape[0], k = 0)]) / 2
            K = (theta ** 2) * jnp.exp( - distances / (2 * length_scale_squared))
            return K, jnp.sqrt(length_scale_squared)
        else:
            K = (theta ** 2) * jnp.exp( - distances / (2 * length_scale ** 2))
            return K, length_scale
        
    def __repr__(self) -> str:
        """
        Return a string representation of the Squared Exponential kernel.

        Returns
        -------
        str
            String representation of the Squared Exponential kernel.
        """
        raise NotImplementedError

    def get_params(self) -> Dict[str, float]:
        """
        Get the parameters of the kernel.

        Returns
        -------
        Dict[str, float]
            Dictionary containing the parameters of the kernel.
        """
        return {'theta': self.theta,
                'length_scale': self.length_scale,}

    def set_params(self, **params):
        """
        Set the parameters of the kernel.
        """
        for param_name, param_value in params.items():
            setattr(self, param_name, param_value)


class ColumnwiseRBF(Kernel):
    """
    Computes the product of Radial Basis Function (RBF) kernels columnwise.

    Args:
        length_scales (Union[float, list]): Length scales for each dimension.
            If float, the same length scale is applied to all dimensions. If list,
            it should contain a length scale for each dimension.
        use_length_scale_heuristic (bool): Whether to use median length scale
            heuristic for RBF kernel.
        use_jit_call (bool): Whether to use Just-In-Time (JIT) compilation for
            computing the kernel.

    Attributes:
        length_scales (Union[float, list]): Length scales for each dimension.
        use_length_scale_heuristic (bool): Whether to use median length scale
            heuristic for RBF kernel.
        use_jit_call (bool): Whether to use Just-In-Time (JIT) compilation for
            computing the kernel.
        baseRBF (RBF): Base RBF kernel instance.

    Returns:
        jnp.ndarray: Kernel matrix computed from the product of RBF kernels.
    """
    def __init__(self,
                 length_scales: Union[float, list] = 0.5,
                 use_length_scale_heuristic: bool = False,
                 length_scale_heuristic_quantile: float = 0.5,
                 use_jit_call: bool = False) -> None:
        """
        Initializes the ColumnwiseRBF kernel.

        Args:
            length_scales (Union[float, list]): Length scales for each dimension.
                If float, the same length scale is applied to all dimensions. If list,
                it should contain a length scale for each dimension.
            use_length_scale_heuristic (bool): Whether to use median length scale
                heuristic for RBF kernel.
            use_jit_call (bool): Whether to use Just-In-Time (JIT) compilation for
                computing the kernel.
        """
        self.length_scales = length_scales
        self.use_length_scale_heuristic = use_length_scale_heuristic
        self.length_scale_heuristic_quantile = length_scale_heuristic_quantile
        self.use_jit_call = use_jit_call    
        self.baseRBF = RBF(use_length_scale_heuristic = use_length_scale_heuristic,
                           length_scale_heuristic_quantile = length_scale_heuristic_quantile,
                           use_jit_call = use_jit_call)
        

    def __call__(self, 
                 X: jnp.ndarray,
                 Y: jnp.ndarray = None) -> jnp.ndarray:
        """
        Computes the product of Radial Basis Function (RBF) kernels columnwise.

        Args:
            X (jnp.ndarray): Input data matrix of shape (n_samples, n_features).
            Y (jnp.ndarray, optional): Another input data matrix of shape (n_samples, n_features).
                If None, Y is assumed to be equal to X.

        Returns:
            jnp.ndarray: Kernel matrix computed from the product of RBF kernels.
        """
        if Y is None:
            Y = X

        row_x_size, column_size = X.shape
        row_y_size, _ = Y.shape
        if isinstance(self.length_scales, float):
            length_scales = [self.length_scales for _ in range(column_size)]
        else:
            length_scales = self.length_scales

        K = jnp.ones((row_x_size, row_y_size))

        for jj in range(column_size):
            X_ = X[:, jj].reshape(-1, 1)
            Y_ = Y[:, jj].reshape(-1, 1)

            if not self.use_length_scale_heuristic:
                self.baseRBF.set_params(**{"length_scale": length_scales[jj]})

            K_ = self.baseRBF(X_, Y_)
            length_scales[jj] = self.baseRBF.length_scale
            K *= K_

        self.length_scales = length_scales
        return K

    def get_params(self) -> Dict[str, float]:
        """
        Get the parameters of the kernel.

        Returns
        -------
        Dict[str, float]
            Dictionary containing the parameters of the kernel.
        """
        return {'length_scales': self.length_scales}

    def set_params(self, **params):
        """
        Set the parameters of the kernel.
        """
        for param_name, param_value in params.items():
            setattr(self, param_name, param_value)


class NuclearRBF(Kernel):

    def __init__(self,
                 length_scale: float = 1.0,
                 eta: float = 1.0) -> None:
        self.length_scale = length_scale
        self.eta = eta

    def __call__(self, 
                 X: jnp.ndarray, 
                 Y: Optional[jnp.ndarray] = None,
                 ) -> jnp.ndarray:
        data_shape = X.shape[1]
        if Y is None:
            Y = X
        X_Y_distances = pairwise_squared_distance(X, Y)
        X_minus_Y_distance_divided_by_4 = pairwise_squared_distance(X / 2.0, -1.0 * Y / 2.0)
        K_1 = jnp.exp( - X_Y_distances / (4.0 * self.length_scale ** 2))
        K_2 = jnp.exp( (- 0.5) * X_minus_Y_distance_divided_by_4 / (0.5 * self.length_scale ** 2 + self.eta ** 2))
        K = K_1 * K_2
        K *= ((2.0 * jnp.pi) ** data_shape) * ((2.0 / (self.length_scale ** 2) + 1 / (self.eta ** 2)) ** (- data_shape / 2.0))
        return K
    
    def __repr__(self) -> str:
        """
        Return a string representation of the RBF kernel.

        Returns
        -------
        str
            String representation of the RBF kernel.
        """
        raise NotImplementedError

    def get_params(self) -> Dict[str, float]:
        """
        Get the parameters of the kernel.

        Returns
        -------
        Dict[str, float]
            Dictionary containing the parameters of the kernel.
        """
        return {'length_scale': self.length_scale,
                'eta': self.eta}

    def set_params(self, **params):
        """
        Set the parameters of the kernel.
        """
        for param_name, param_value in params.items():
            setattr(self, param_name, param_value)


class FourthOrderGaussianKernel(Kernel):

    """
    Parameters
    ----------
    length_scale : float, optional
        The length scale parameter of the RBF kernel. It controls the smoothness of the 
        resulting function. Default is 0.5.
    use_length_scale_heuristic : bool, optional
        If True, the length scale is set to the median pairwise distance of the input data, 
        divided by the square root of 2. This heuristic can be useful when the length scale 
        is not known a priori. Default is False.
    columnwise : bool, optional
        If True, the kernel function is computed column-wise. This affects the computation 
        of the kernel matrix. Default is True.

    Attributes
    ----------
    length_scale : float
        The length scale parameter of the RBF kernel.
    use_length_scale_heuristic : bool
        Indicates whether the median length scale heuristic is used.
    columnwise : bool
        Indicates whether the kernel is computed column-wise.

    Methods
    -------
    __call__(X: jnp.ndarray, Y: Optional[jnp.ndarray] = None) -> jnp.ndarray
        Computes the kernel matrix between input arrays X and Y. If Y is not provided, 
        the kernel matrix is computed between X and itself.

    Examples
    --------
    >>> kernel = FourthOrderGaussianKernel(length_scale=1.0)
    >>> X = jnp.array([[1, 2], [3, 4]])
    >>> K = kernel(X)
    """

    def __init__(self, 
                 length_scale: float = 0.5,
                 use_length_scale_heuristic: bool = False,
                 columnwise = True) -> None:
        
        self.length_scale = length_scale
        self.use_length_scale_heuristic = use_length_scale_heuristic
        self.columnwise = columnwise

    def __call__(self, 
                 X: jnp.ndarray, 
                 Y: Optional[jnp.ndarray] = None,
                 ) -> jnp.ndarray:
        """
        Computes the kernel matrix between input arrays X and Y using the fourth-order Gaussian kernel.

        Parameters
        ----------
        X : jnp.ndarray
            Input data array of shape (n_samples_X, n_features).
        Y : jnp.ndarray, optional
            Input data array of shape (n_samples_Y, n_features). If not provided, the 
            kernel matrix is computed between X and itself.

        Returns
        -------
        jnp.ndarray
            The computed kernel matrix of shape (n_samples_X, n_samples_Y).
        """

        if Y is None:
            Y = X

        distances = pairwise_squared_distance(X, Y)
        if self.use_length_scale_heuristic:
            length_scale_squared = jnp.median(distances[jnp.triu_indices(distances.shape[0], k = 0)]) / 2
            self.length_scale = jnp.sqrt(length_scale_squared)
        
        diff_data = X[:, jnp.newaxis, :] - Y[jnp.newaxis, :, :]
        u = diff_data / self.length_scale
        if self.columnwise:
            kernel_tensor = jnp.exp(- u ** 2 / 2.0) * (3 - u ** 2) / 2.0 #/ jnp.sqrt(6.28)
            K = jnp.product(kernel_tensor, axis = 2)
        else:
            K = jnp.exp(- (u ** 2).sum(-1) / 2.0) * (3 - (u ** 2).sum(-1)) / 2.0
        return K

    def __repr__(self) -> str:
        """
        Return a string representation of the RBF kernel.

        Returns
        -------
        str
            String representation of the RBF kernel.
        """
        raise NotImplementedError

    def get_params(self) -> Dict[str, float]:
        """
        Get the parameters of the kernel.

        Returns
        -------
        Dict[str, float]
            Dictionary containing the parameters of the kernel.
        """
        return {'length_scale': self.length_scale}

    def set_params(self, **params):
        """
        Set the parameters of the kernel.
        """
        for param_name, param_value in params.items():
            setattr(self, param_name, param_value)


class EpanechnikovKernel(Kernel):

    def __init__(self, 
                 length_scales: Union[float, List[float]] = 0.5,
                 use_length_scale_heuristic: bool = True,
                 columnwise: bool = True,
                 c: float = 1.0):
        """
        Initializes the Epanechnikov kernel.

        Args:
            length_scales (Union[float, List[float]]): The length scales for the kernel. 
                Can be a single float or a list of floats. Defaults to 0.5.
            use_length_scale_heuristic (bool): Whether to use the length scale heuristic. Defaults to True.
            columnwise (bool): Whether to compute the kernel columnwise. Defaults to True.
            c (float): Scaling constant for the length scale heuristic. Defaults to 1.0.
        """
        self.length_scales = length_scales
        self.use_length_scale_heuristic = use_length_scale_heuristic
        self.columnwise = columnwise
        self.c = c

    def __call__(self, 
                 X: jnp.ndarray, 
                 Y: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Computes the product of Epanechnikov kernels either columnwise or based on the norms of the rows.

        Args:
            X (jnp.ndarray): Input data matrix of shape (n_samples, n_features).
            Y (Optional[jnp.ndarray]): Another input data matrix of shape (n_samples, n_features). 
                If None, Y is assumed to be equal to X.

        Returns:
            jnp.ndarray: Kernel matrix computed from the product of Epanechnikov kernels.
        """
        if Y is None:
            Y = X
        n_data, columnsize = X.shape
        if not self.columnwise:
            X = ((X ** 2).sum(-1)).reshape(-1, 1)
            Y = ((Y ** 2).sum(-1)).reshape(-1, 1)

        if self.use_length_scale_heuristic:
            self.length_scales = self.c * (jnp.std(X, axis = 0) / (n_data ** 0.25)).reshape(1, 1, -1)
        else:
            if isinstance(self.length_scales, float):
                self.length_scales = jnp.array([self.length_scales for _ in range(columnsize)]).reshape(1, 1, -1)

        distances = ((X[:, None] - Y[None, :]) ** 2) / self.length_scales
        K = jnp.maximum((1.0 - distances) * (3 / 4) / self.length_scales, 0)
        K = jnp.prod(K, axis = -1)
        return K

    def get_params(self) -> Dict[str, float]:
        """
        Get the parameters of the kernel.

        Returns
        -------
        Dict[str, float]
            Dictionary containing the parameters of the kernel.
        """
        return {'length_scales': self.length_scales}

    def set_params(self, **params):
        """
        Set the parameters of the kernel.
        """
        for param_name, param_value in params.items():
            setattr(self, param_name, param_value)


class MaternKernel(Kernel):

    def __init__(self,
                 p: int,
                 length_scale: float = 1.0,
                 use_length_scale_heuristic: bool = False) -> None:
        """
        Initialize the Matern kernel with the given parameters.

        Parameters:
        p (int): The smoothness parameter of the Matern kernel.
        length_scale (float): The length scale parameter of the Matern kernel.
        use_length_scale_heuristic (bool): Whether to use the median length scale heuristic.
        """
        self.p = p
        self.length_scale = length_scale
        self.use_length_scale_heuristic = use_length_scale_heuristic

        dist = sympy.symbols("d")
        length_scale_squared = sympy.symbols("l2")
        matern = 0.0
        for i in range(p + 1):
            const = self.log_factorial(p + i) - self.log_factorial(p - i) - self.log_factorial(i)
            const = const + self.log_factorial(p) - self.log_factorial(2 * p)
            const = jnp.exp(const)
            matern = matern + const * (2 * np.sqrt(2 * p + 1) * dist / length_scale_squared)**(p - i)
        matern = matern * sympy.exp( - np.sqrt(2 * p + 1) * dist / length_scale_squared)
        f_matern = sympy.utilities.lambdify([dist, length_scale_squared], matern, ["jax"])
        self.f_matern = f_matern
        
    def log_factorial(self, p: int) -> float:
        """
        Compute the logarithm of the factorial of a given number.

        Parameters:
        p (int): The number for which to compute the factorial.

        Returns:
        float: The logarithm of the factorial of p.
        """
        return loggamma(p + 1)

    def __call__(self, 
                 X: jnp.ndarray,
                 Y: Optional[jnp.ndarray] = None,) -> jnp.ndarray:
        """
        Compute the Matern kernel between two sets of input points.

        Parameters:
        X (jnp.ndarray): A set of input points.
        Y (jnp.ndarray, optional): Another set of input points. If None, Y is set to X.

        Returns:
        jnp.ndarray: The computed Matern kernel matrix.
        """
        
        if Y is None:
            Y = X
        distances = jnp.sqrt(pairwise_squared_distance(X, Y))
        if self.use_length_scale_heuristic:
            # length_scale = jnp.median(distances[jnp.triu_indices(distances.shape[0], k = 1)]).item() 
            length_scale = jnp.median(distances).item() 
            self.length_scale = length_scale
        else:
            length_scale = self.length_scale

        K = self.f_matern(distances, length_scale)
        return K

    def __repr__(self) -> str:
        """
        Return a string representation of the RBF kernel.

        Returns
        -------
        str
            String representation of the RBF kernel.
        """
        raise NotImplementedError

    def get_params(self) -> Dict[str, float]:
        """
        Get the parameters of the kernel.

        Returns
        -------
        Dict[str, float]
            Dictionary containing the parameters of the kernel.
        """
        return {'length_scale': self.length_scale}

    def set_params(self, **params):
        """
        Set the parameters of the kernel.
        """
        for param_name, param_value in params.items():
            setattr(self, param_name, param_value)


class ColumnwiseMaternKernel(Kernel):

    def __init__(self,
                 p: Union[float, list] = 0,
                 length_scales: Union[float, list] = 0.5,
                 use_length_scale_heuristic: bool = False,) -> None:

        self.p = p
        self.length_scales = length_scales
        self.use_length_scale_heuristic = use_length_scale_heuristic 
        self.baseMatern = MaternKernel(p, use_length_scale_heuristic = use_length_scale_heuristic,)
        

    def __call__(self, 
                 X: jnp.ndarray,
                 Y: jnp.ndarray = None) -> jnp.ndarray:
        """
        Computes the product of Matern kernels columnwise.

        Args:
            X (jnp.ndarray): Input data matrix of shape (n_samples, n_features).
            Y (jnp.ndarray, optional): Another input data matrix of shape (n_samples, n_features).
                If None, Y is assumed to be equal to X.

        Returns:
            jnp.ndarray: Kernel matrix computed from the product of RBF kernels.
        """
        if Y is None:
            Y = X

        row_x_size, column_size = X.shape
        row_y_size, _ = Y.shape
        if isinstance(self.length_scales, float):
            length_scales = [self.length_scales for _ in range(column_size)]
        else:
            length_scales = self.length_scales

        K = jnp.ones((row_x_size, row_y_size))

        for jj in range(column_size):
            X_ = X[:, jj].reshape(-1, 1)
            Y_ = Y[:, jj].reshape(-1, 1)

            if not self.use_length_scale_heuristic:
                self.baseMatern.set_params(**{"length_scale": length_scales[jj]})

            K_ = self.baseMatern(X_, Y_)
            length_scales[jj] = self.baseMatern.length_scale
            K *= K_

        self.length_scales = length_scales
        return K

    def get_params(self) -> Dict[str, float]:
        """
        Get the parameters of the kernel.

        Returns
        -------
        Dict[str, float]
            Dictionary containing the parameters of the kernel.
        """
        return {'length_scales': self.length_scales}

    def set_params(self, **params):
        """
        Set the parameters of the kernel.
        """
        for param_name, param_value in params.items():
            setattr(self, param_name, param_value)


class LaplacianKernel(Kernel):
    """
    Laplacian kernel.

    Parameters
    ----------
    length_scale : float, optional
        The length scale parameter of the Laplacian kernel.

    Attributes
    ----------
    length_scale : float
        The length scale parameter of the Laplacian kernel.
    """
    def __init__(self, 
                 length_scale: float = 0.5,
                 use_length_scale_heuristic: bool = False,
                 length_scale_heuristic_quantile: float = 0.5,
                 use_jit_call: bool = False) -> None:
        self.length_scale = length_scale
        self.use_length_scale_heuristic = use_length_scale_heuristic
        self.length_scale_heuristic_quantile = length_scale_heuristic_quantile
        self.use_jit_call = use_jit_call    
        

    def __call__(self, 
                 X: jnp.ndarray, 
                 Y: Optional[jnp.ndarray] = None,
                 ) -> jnp.ndarray:
        """
        Return the Laplacian kernel k(X, Y) and (optionally its gradient will be added later in the future)

        Parameters
        ----------
        X : jnp.ndarray
            Left argument of the returned kernel k(X, Y).
        Y : jnp.ndarray, optional
            Right argument of the returned kernel k(X, Y). If None, k(X, X) is evaluated instead.

        Returns
        -------
        jnp.ndarray
            Kernel k(X, Y)
        """
        if Y is None:
            Y = X

        if self.use_jit_call:
            K, length_scale = self.call(X, Y, self.length_scale, self.use_length_scale_heuristic, self.length_scale_heuristic_quantile)
            self.length_scale = float(length_scale)
            return K
        else:
            distances = pairwise_absolute_distance(X, Y)
            if self.use_length_scale_heuristic:
                # length_scale_squared = jnp.median(distances[jnp.triu_indices(distances.shape[0], k = 1)]) / 2.0
                length_scale_squared = jnp.quantile(distances[jnp.triu_indices(distances.shape[0], k = 1)], self.length_scale_heuristic_quantile) / 2.0
                K = jnp.exp( - distances / (2 * length_scale_squared))
                self.length_scale = float(jnp.sqrt(length_scale_squared))
            else:
                K = jnp.exp( - distances / (2 * self.length_scale ** 2))
            return K
        
    @staticmethod
    @partial(jit, static_argnums=3) 
    def call(X: jnp.ndarray, 
             Y: Optional[jnp.ndarray] = None,
             length_scale: float = 1.0,
             use_length_scale_heuristic: Optional[bool] = False,
             length_scale_heuristic_quantile: float = 0.5) -> jnp.ndarray:
        """
        Compute the Laplacian kernel k(X, Y).

        Parameters
        ----------
        X : jnp.ndarray
            Left argument of the kernel.
        Y : jnp.ndarray, optional
            Right argument of the kernel. If None, k(X, X) is evaluated.
        length_scale : float, optional
            The length scale parameter of the Laplacian kernel.

        Returns
        -------
        jnp.ndarray
            Computed Laplacian kernel k(X, Y)
        """
        if Y is None:
            Y = X
        distances = pairwise_absolute_distance(X, Y)
        if use_length_scale_heuristic:
            # length_scale_squared = jnp.median(distances[jnp.triu_indices(distances.shape[0], k = 1)]) / 2
            length_scale_squared = jnp.quantile(distances[jnp.triu_indices(distances.shape[0], k = 1)], length_scale_heuristic_quantile) / 2.0
            K = jnp.exp( - distances / (2 * length_scale_squared))
            return K, jnp.sqrt(length_scale_squared)
        else:
            K = jnp.exp( - distances / (2 * length_scale ** 2))
            return K, length_scale
        
    def __repr__(self) -> str:
        """
        Return a string representation of the RBF kernel.

        Returns
        -------
        str
            String representation of the RBF kernel.
        """
        raise NotImplementedError

    def get_params(self) -> Dict[str, float]:
        """
        Get the parameters of the kernel.

        Returns
        -------
        Dict[str, float]
            Dictionary containing the parameters of the kernel.
        """
        return {'length_scale': self.length_scale}

    def set_params(self, **params):
        """
        Set the parameters of the kernel.
        """
        for param_name, param_value in params.items():
            setattr(self, param_name, param_value)

    def tree_flatten(self) -> Tuple[Tuple, Dict[str, Any]]:
        children = ()
        aux_data = {"length_scale": self.length_scale}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls, aux_data: Dict[str, Any], children: Tuple
    ):
        return cls(*children, **aux_data)


class ColumnwiseLaplacianKernel(Kernel):
    """
    Computes the product of Laplacian kernels columnwise.

    Args:
        length_scales (Union[float, list]): Length scales for each dimension.
            If float, the same length scale is applied to all dimensions. If list,
            it should contain a length scale for each dimension.
        use_length_scale_heuristic (bool): Whether to use median length scale
            heuristic for Laplacian kernel.
        use_jit_call (bool): Whether to use Just-In-Time (JIT) compilation for
            computing the kernel.

    Attributes:
        length_scales (Union[float, list]): Length scales for each dimension.
        use_length_scale_heuristic (bool): Whether to use median length scale
            heuristic for Laplacian kernel.
        use_jit_call (bool): Whether to use Just-In-Time (JIT) compilation for
            computing the kernel.
        baseLaplacian (Laplacian): Base Laplacian kernel instance.

    Returns:
        jnp.ndarray: Kernel matrix computed from the product of Laplacian kernels.
    """
    def __init__(self,
                 length_scales: Union[float, list] = 0.5,
                 use_length_scale_heuristic: bool = False,
                 length_scale_heuristic_quantile: float = 0.5,
                 use_jit_call: bool = False) -> None:
        """
        Initializes the ColumnwiseLaplacianKernel class.

        Args:
            length_scales (Union[float, list]): Length scales for each dimension.
                If float, the same length scale is applied to all dimensions. If list,
                it should contain a length scale for each dimension.
            use_length_scale_heuristic (bool): Whether to use median length scale
                heuristic for Laplacian kernel.
            use_jit_call (bool): Whether to use Just-In-Time (JIT) compilation for
                computing the kernel.
        """
        self.length_scales = length_scales
        self.use_length_scale_heuristic = use_length_scale_heuristic
        self.length_scale_heuristic_quantile = length_scale_heuristic_quantile
        self.use_jit_call = use_jit_call    
        self.baseLaplacian = LaplacianKernel(use_length_scale_heuristic = use_length_scale_heuristic,
                                             length_scale_heuristic_quantile = length_scale_heuristic_quantile,
                                             use_jit_call = use_jit_call)
        

    def __call__(self, 
                 X: jnp.ndarray,
                 Y: jnp.ndarray = None) -> jnp.ndarray:
        """
        Computes the product of Radial Basis Function (Laplacian) kernels columnwise.

        Args:
            X (jnp.ndarray): Input data matrix of shape (n_samples, n_features).
            Y (jnp.ndarray, optional): Another input data matrix of shape (n_samples, n_features).
                If None, Y is assumed to be equal to X.

        Returns:
            jnp.ndarray: Kernel matrix computed from the product of Laplacian kernels.
        """
        if Y is None:
            Y = X

        row_x_size, column_size = X.shape
        row_y_size, _ = Y.shape
        if isinstance(self.length_scales, float):
            length_scales = [self.length_scales for _ in range(column_size)]
        else:
            length_scales = self.length_scales

        K = jnp.ones((row_x_size, row_y_size))

        for jj in range(column_size):
            X_ = X[:, jj].reshape(-1, 1)
            Y_ = Y[:, jj].reshape(-1, 1)

            if not self.use_length_scale_heuristic:
                self.baseLaplacian.set_params(**{"length_scale": length_scales[jj]})

            K_ = self.baseLaplacian(X_, Y_)
            length_scales[jj] = self.baseLaplacian.length_scale
            K *= K_

        self.length_scales = length_scales
        return K

    def get_params(self) -> Dict[str, float]:
        """
        Get the parameters of the kernel.

        Returns
        -------
        Dict[str, float]
            Dictionary containing the parameters of the kernel.
        """
        return {'length_scales': self.length_scales}

    def set_params(self, **params):
        """
        Set the parameters of the kernel.
        """
        for param_name, param_value in params.items():
            setattr(self, param_name, param_value)
