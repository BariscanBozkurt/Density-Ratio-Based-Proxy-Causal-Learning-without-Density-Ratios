import numpy as np
import jax.numpy as jnp
from jax import jit
from jax.scipy.optimize import minimize as jax_minimize
from sklearn.base import BaseEstimator, RegressorMixin

import sys
sys.path.append("..")

from utils.kernel_utils import RBF
from utils.linalg_utils import make_psd, cartesian_product

from typing import Callable, Tuple, Optional, Union, Dict

from jax import config
config.update("jax_enable_x64", True)

#### THE FOLLOWING THERE PYTHON CLASSES ARE THE IMPLEMENTATION OF CAUSAL FUNCTION ESTIMATION ALGORITHMS FROM THE FOLLOWING PAPER:
#### R Singh, L Xu, and A Gretton. Kernel methods for causal functions: dose, heterogeneous and incremental response
#### curves. Biometrika, 111(2):497–516, 07 2023. ISSN 1464-3510. doi: 10.1093/biomet/asad042. URL https:
#### //doi.org/10.1093/biomet/asad042.


class KernelATE(BaseEstimator, RegressorMixin):

    def __init__(self,
                 kernel_A: Callable,
                 kernel_X: Callable, 
                 lambda_: float = 1e-3,
                 optimize_regularization_parameters: bool = True,
                 lambda_optimization_range: Tuple[float, float] = (1e-9, 1.0),
                 **kwargs) -> None:
        """
        Initialize the KernelATE estimator.

        Parameters:
        - kernel_A (Callable): Kernel function for variable A.
        - kernel_X (Callable): Kernel function for variable X.
        - lambda_ (float, optional): Regularization parameter. Defaults to 1e-3.
        - optimize_regularization_parameters (bool, optional): Flag to optimize regularization parameters. Defaults to True.
        - lambda_optimization_range (Tuple[float, float], optional): Range for lambda optimization. Defaults to (1e-9, 1.0).
        - **kwargs: Additional parameters.
        """
        # super().__init__()

        kernel_X_params = kwargs.pop('kernel_X_params', None)
        kernel_A_params = kwargs.pop('kernel_A_params', None)
        regularization_grid_points = kwargs.pop('regularization_grid_points', 150)

        if (not isinstance(kernel_X, Callable)):
            raise Exception("Kernel for X must be callable")
        if (not isinstance(kernel_A, Callable)):
            raise Exception("Kernel for A must be callable")
            
        self.kernel_X = kernel_X
        self.kernel_A = kernel_A
        if kernel_X_params is not None:
            self.kernel_X.set_params(**kernel_X_params)
        if kernel_A_params is not None:
            self.kernel_A.set_params(**kernel_A_params)

        self.lambda_ = lambda_
        self.optimize_regularization_parameters = optimize_regularization_parameters
        self.lambda_optimization_range = lambda_optimization_range
        self.regularization_grid_points = regularization_grid_points

    @staticmethod
    @jit
    def ridge_penalty_loss(lambda_: float, 
                           K_WW: jnp.ndarray, 
                           Y: jnp.ndarray) -> float:
        """
        Compute the ridge penalty loss.

        Parameters:
        - lambda_ (float): Regularization parameter.
        - K_WW (jnp.ndarray): Kernel matrix for variable W.
        - Y (jnp.ndarray): Target values.

        Returns:
        - float: Ridge penalty loss.
        """
        n = K_WW.shape[0]
        identity = jnp.eye(n)
        # H_alpha = identity - make_psd(K_WW) @ jnp.linalg.inv(make_psd(K_WW) + n * jnp.exp(log_alpha) * identity)
        H_alpha = identity - jnp.linalg.solve((make_psd(K_WW) + n * lambda_ * identity).T, make_psd(K_WW).T).T
        H_tilde_alpha_inv = jnp.diag(1/jnp.diag(H_alpha))
        loss = (jnp.linalg.norm(H_tilde_alpha_inv @ H_alpha @ Y) ** 2) # /n
        return loss
    
    def fit(self, 
            AX: Tuple[jnp.ndarray, jnp.ndarray], 
            Y: jnp.ndarray) -> None:
        """
        Fit the KernelATE model.

        Parameters:
        - AX (Tuple[jnp.ndarray, jnp.ndarray]): Tuple of data arrays (A, X).
        - Y (jnp.ndarray): Target values.
        """
        A, X = AX
        n = A.shape[0]

        kernel_A = self.kernel_A
        kernel_X = self.kernel_X
        lambda_optimization_range = self.lambda_optimization_range
        regularization_grid_points = self.regularization_grid_points
        lambda_ = self.lambda_

        K_XX = kernel_X(X)
        K_AA = kernel_A(A)

        if hasattr(self.kernel_A, 'use_length_scale_heuristic'):
            self.kernel_A.use_length_scale_heuristic = False
        if hasattr(self.kernel_X, 'use_length_scale_heuristic'):
            self.kernel_X.use_length_scale_heuristic = False
        
        if self.optimize_regularization_parameters:
            K_WW = K_XX * K_AA
            lambda_list = jnp.logspace(jnp.log(lambda_optimization_range[0]), jnp.log(lambda_optimization_range[1]), regularization_grid_points, base = jnp.exp(1))
            ridge_penaly_loss_list = jnp.array([self.ridge_penalty_loss(lambda_, K_WW, Y) for lambda_ in lambda_list])
            lambda_ = lambda_list[jnp.argmin(ridge_penaly_loss_list).item()]
            self.lambda_ = lambda_
        K_WW_with_ridge = K_XX * K_AA + n * self.lambda_ * jnp.eye(n)
        K_XX_mean = jnp.mean(K_XX, axis = 0)

        self.K_WW_with_ridge = K_WW_with_ridge
        self.K_XX_mean = K_XX_mean
        self.A_train = A
        self.Y = Y

    def predict(self, 
                A: jnp.ndarray) -> jnp.ndarray:
        """
        Predict outcomes for new data points.

        Parameters:
        - A (jnp.ndarray): New data points for variable A.

        Returns:
        - jnp.ndarray: Predicted values.
        """
        kernel_A = self.kernel_A
        
        K_WW_with_ridge = self.K_WW_with_ridge
        K_XX_mean = self.K_XX_mean
        A_train = self.A_train
        Y = self.Y

        K_Aa = kernel_A(A_train, A)
        K_Aa_times_K_XX_mean = K_Aa * K_XX_mean[:, jnp.newaxis]
        Y_a_pred = jnp.linalg.solve(K_WW_with_ridge, K_Aa_times_K_XX_mean).T @ Y
        return Y_a_pred

    def fit_predict(self, 
                    AX: Tuple[jnp.ndarray, jnp.ndarray], 
                    Y: jnp.ndarray) -> jnp.ndarray:
        """
        Fit the model and predict outcomes for the training data.

        Parameters:
        - AX (Tuple[jnp.ndarray, jnp.ndarray]): Tuple of data arrays (A, X).
        - Y (jnp.ndarray): Target values.

        Returns:
        - jnp.ndarray: Predicted values.
        """
        self.fit(AX, Y)
        A, _ = AX
        return self.predict(A)
    


class KernelATT(BaseEstimator, RegressorMixin):

    def __init__(self,
                 kernel_A: Callable,
                 kernel_X: Callable, 
                 lambda_: float = 1e-3,
                 lambda2_: float = 1e-3,
                 optimize_regularization_parameters: bool = True,
                 lambda_optimization_range: Tuple[float, float] = (1e-9, 1.0),
                 **kwargs) -> None:
        """
        Initialize the KernelATT estimator.

        Parameters:
        - kernel_A (Callable): Kernel function for variable A.
        - kernel_X (Callable): Kernel function for variable X.
        - lambda_ (float, optional): Regularization parameter. Defaults to 1e-3.
        - lambda2_ (float, optional): Regularization parameter. Defaults to 1e-3.
        - optimize_regularization_parameters (bool, optional): Flag to optimize regularization parameters. Defaults to True.
        - lambda_optimization_range (Tuple[float, float], optional): Range for lambda optimization. Defaults to (1e-9, 1.0).
        - **kwargs: Additional parameters.
        """
        # super().__init__()

        kernel_X_params = kwargs.pop('kernel_X_params', None)
        kernel_A_params = kwargs.pop('kernel_A_params', None)
        regularization_grid_points = kwargs.pop('regularization_grid_points', 150)

        if (not isinstance(kernel_X, Callable)):
            raise Exception("Kernel for X must be callable")
        if (not isinstance(kernel_A, Callable)):
            raise Exception("Kernel for A must be callable")
            
        self.kernel_X = kernel_X
        self.kernel_A = kernel_A
        if kernel_X_params is not None:
            self.kernel_X.set_params(**kernel_X_params)
        if kernel_A_params is not None:
            self.kernel_A.set_params(**kernel_A_params)

        self.lambda_ = lambda_
        self.lambda2_ = lambda2_
        self.optimize_regularization_parameters = optimize_regularization_parameters
        self.lambda_optimization_range = lambda_optimization_range
        self.regularization_grid_points = regularization_grid_points

    @staticmethod
    @jit
    def ridge_penalty_loss(lambda_: float, 
                           K_WW: jnp.ndarray, 
                           Y: jnp.ndarray) -> float:
        """
        Compute the ridge penalty loss.

        Parameters:
        - lambda_ (float): Regularization parameter.
        - K_WW (jnp.ndarray): Kernel matrix for variable W.
        - Y (jnp.ndarray): Target values.

        Returns:
        - float: Ridge penalty loss.
        """
        n = K_WW.shape[0]
        identity = jnp.eye(n)
        # H_alpha = identity - make_psd(K_WW) @ jnp.linalg.inv(make_psd(K_WW) + n * jnp.exp(log_alpha) * identity)
        H_alpha = identity - jnp.linalg.solve((make_psd(K_WW) + n * lambda_ * identity).T, make_psd(K_WW).T).T
        H_tilde_alpha_inv = jnp.diag(1/jnp.diag(H_alpha))
        loss = (jnp.linalg.norm(H_tilde_alpha_inv @ H_alpha @ Y) ** 2) # /n
        return loss
    
    @staticmethod
    @jit
    def conditional_mean_embedding_regularization_loss(lambda_, K_AA, K_YY):
        """
        See algorithm 7 in https://arxiv.org/abs/2012.10315
        Kernel Methods for Unobserved Confounding: Negative Controls, Proxies, and Instruments by Rahul Singh

        Compute the conditional mean embedding regularization loss.

        Parameters:
        - lambda_ (float): Regularization parameter.
        - K_AA (jnp.ndarray): Kernel matrix for variable A.
        - K_YY (jnp.ndarray): Kernel matrix for target values Y.

        Returns:
        - float: Conditional mean embedding regularization loss.
        """
        n = K_AA.shape[0]
        R = K_AA @ jnp.linalg.inv(make_psd(K_AA) + n * lambda_ * jnp.eye(n))
        S = jnp.diag((1 / (1 - jnp.diag(R))) ** 2)
        T = S @ (K_YY - 2 * K_YY @ R.T + R @ K_YY @ R.T)
        cost = jnp.trace(T)
        return cost
    
    def fit(self, 
            AX: Tuple[jnp.ndarray, jnp.ndarray], 
            Y: jnp.ndarray) -> None:
        """
        Fit the KernelATT model.

        Parameters:
        - AX (Tuple[jnp.ndarray, jnp.ndarray]): Tuple of data arrays (A, X).
        - Y (jnp.ndarray): Target values.
        """
        A, X = AX
        n = A.shape[0]

        kernel_A = self.kernel_A
        kernel_X = self.kernel_X
        lambda_optimization_range = self.lambda_optimization_range
        regularization_grid_points = self.regularization_grid_points
        lambda_, lambda2_ = self.lambda_, self.lambda2_
        
        K_XX = kernel_X(X)
        K_AA = kernel_A(A)

        if hasattr(self.kernel_A, 'use_length_scale_heuristic'):
            self.kernel_A.use_length_scale_heuristic = False
        if hasattr(self.kernel_X, 'use_length_scale_heuristic'):
            self.kernel_X.use_length_scale_heuristic = False
        
        if self.optimize_regularization_parameters:
            K_WW = K_XX * K_AA
            lambda_list = jnp.logspace(jnp.log(lambda_optimization_range[0]), jnp.log(lambda_optimization_range[1]), regularization_grid_points, base = jnp.exp(1))
            ridge_penaly_loss_list = jnp.array([self.ridge_penalty_loss(lambda_, K_WW, Y) for lambda_ in lambda_list])
            lambda_ = lambda_list[jnp.argmin(ridge_penaly_loss_list).item()]
            self.lambda_ = lambda_

            cme_penalty_loss_list = jnp.array([self.conditional_mean_embedding_regularization_loss(lambda2_, K_AA, K_XX) for lambda2_ in lambda_list])
            lambda2_ = lambda_list[jnp.argmin(cme_penalty_loss_list).item()]
            self.lambda2_ = lambda2_

        self.ridge_weights = Y.T @ jnp.linalg.inv(make_psd(K_AA * K_XX) + n * lambda_ * jnp.eye(n))
        self.cme_weights = K_XX @ jnp.linalg.inv(make_psd(K_AA) + n * lambda2_ * jnp.eye(n))
        self.A_train = A

    def predict(self, A: jnp.ndarray, a_prime: jnp.ndarray) -> jnp.ndarray:
        """
        Predict using the fitted KernelATT model.

        Parameters:
        - A (jnp.ndarray): Input data for variable A.
        - a_prime (jnp.ndarray): Input data for variable a'.

        Returns:
        - jnp.ndarray: Predicted values.
        """
        
        a_prime = jnp.array(a_prime).reshape(-1, 1)
        kernel_A = self.kernel_A

        K_Aa = kernel_A(self.A_train, A)
        K_Aaprime = kernel_A(self.A_train, a_prime)
        if a_prime.shape[0] == 1:
            K_Aaprime = jnp.tile(K_Aaprime, A.shape[0])
        pred = (self.ridge_weights @ (K_Aa * (self.cme_weights @ K_Aaprime))).T
        return pred
