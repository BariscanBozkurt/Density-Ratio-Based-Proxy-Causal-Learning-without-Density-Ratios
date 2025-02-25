import sys
sys.path.append("../../src")

import os
from time import time
import numpy as np
import pandas as pd
import jax.numpy as jnp

from utils.kernel_utils import RBF, ColumnwiseRBF
from causal_models.proxy_causal_learning import KernelAlternativeProxyATT, KernelNegativeControlATT
from causal_models.causal_learning import KernelATT
from generate_experiment_data import generate_synthetic_data
from utils.ml_utils import data_transform

if not os.path.exists("../Results"):
    os.mkdir("../Results")

data_size_list = [2000]
seed_list = np.arange(0, 3000, 100)
a_prime_list = [-1.0, -0.5, 0.25, 0.5, 1.5]
label_variance_in_eta_opt_list = [1.0]
scale_data = True
kernelW_length_scale_list = [0.5, 1.0, 1.5, 2.0, 'median_heur']
df_results = pd.DataFrame(columns = ["Algorithm", "Data_Size", "Seed", "a_prime", "sigma_square", "alternative_proxy_eta_parameter", "ATT_Estimation", "kernelW_length_scale", "Algo_Run_Time"])

#######################################################
#        Alternative Kernel Proxy                     #
#######################################################

for n_plus_m in data_size_list:
    for seed_ in seed_list:
            # np.random.seed(seed_)
            # U, W, Z, A, Y, do_A, EY_do_A = generate_synthetic_data(size = n_plus_m, seed = seed_)
            # W_train, Z_train, A_train, Y_train, do_A, EY_do_A = jnp.array(W), jnp.array(Z), jnp.array(A), jnp.array(Y), jnp.array(do_A), jnp.array(EY_do_A)
            for a_prime in a_prime_list:
                for label_variance_in_eta_opt_ in label_variance_in_eta_opt_list:
                    for kernelW_length_scale_ in kernelW_length_scale_list:
                        np.random.seed(seed_)
                        U, W, Z, A, Y, do_A, EY_do_A = generate_synthetic_data(size = n_plus_m, seed = seed_)
                        W_train, Z_train, A_train, Y_train, do_A, EY_do_A = jnp.array(W), jnp.array(Z), jnp.array(A), jnp.array(Y), jnp.array(do_A), jnp.array(EY_do_A)
    
                        if scale_data:
                            A_transformed, A_transformer = data_transform(A)
                            Z_transformed, Z_transformer = data_transform(Z)
                            W_transformed, W_transformer = data_transform(W)
                            Y_transformed, Y_transformer = data_transform(Y)
    
                            data_size = A_transformed.shape[0]
                            A_transformed = jnp.array(A_transformed).reshape(data_size, -1)
                            Z_transformed = jnp.array(Z_transformed).reshape(data_size, -1)
                            W_transformed = jnp.array(W_transformed).reshape(data_size, -1)
                            Y_transformed = jnp.array(Y_transformed).reshape(data_size, -1)
    
                        t0 = time()
                        RBF_Kernel_Z = RBF(use_length_scale_heuristic = True, use_jit_call = True)
                        RBF_Kernel_A = RBF(use_length_scale_heuristic = True, use_jit_call = True)
                        if kernelW_length_scale_ == 'median_heur':
                            RBF_Kernel_W = ColumnwiseRBF(use_length_scale_heuristic = True, use_jit_call = True)
                        else:
                            RBF_Kernel_W = ColumnwiseRBF(use_length_scale_heuristic = False, length_scales =  kernelW_length_scale_, use_jit_call = True)
    
                        lambda_ = 1e-2
                        eta = 2*1e-3
                        lambda2_ = 1e-3
                        optimize_lambda_parameters = True
                        optimize_eta_parameter = True
                        optimize_zeta_parameter = True
                        lambda_optimization_range = (1e-7, 1.0)
                        zeta_optimization_range = (1e-7, 1.0)
                        eta_optimization_range = (1e-7, 1.0)
                        stage1_perc = 0.5
                        regularization_grid_points = 150
    
    
                        model = KernelAlternativeProxyATT(
                                                            kernel_A = RBF_Kernel_A,
                                                            kernel_W = RBF_Kernel_W, 
                                                            kernel_Z = RBF_Kernel_Z,
                                                            lambda_ = lambda_,
                                                            eta = eta,
                                                            lambda2_ = lambda2_,
                                                            optimize_lambda_parameters = optimize_lambda_parameters,
                                                            optimize_eta_parameter = optimize_eta_parameter,
                                                            optimize_zeta_parameter = optimize_zeta_parameter,
                                                            lambda_optimization_range = lambda_optimization_range,
                                                            zeta_optimization_range = zeta_optimization_range,
                                                            eta_optimization_range = eta_optimization_range,
                                                            stage1_perc = stage1_perc,
                                                            regularization_grid_points = regularization_grid_points, 
                                                            label_variance_in_eta_opt = label_variance_in_eta_opt_,
                                                         )
                        do_A_size = do_A.shape[0]
                        if scale_data:
                            model.fit((A_transformed, W_transformed, Z_transformed), Y_transformed, a_prime)
                            do_A_transformed = (A_transformer.transform(do_A)).reshape(do_A_size, -1)
                            f_struct_pred_transformed = model.predict(do_A_transformed)
                            f_struct_pred = Y_transformer.inverse_transform(f_struct_pred_transformed.reshape(do_A_size, -1)).reshape(do_A_size, -1)
                        else:
                            model.fit((A_train, W_train, Z_train), Y_train, a_prime)
                            f_struct_pred = model.predict(do_A).reshape(do_A_size, -1)
    
                        f_struct_pred = np.array(f_struct_pred)
                        t1 = time()
                        algo_run_time = t1 - t0
    
                        Kernel_Alternative_Proxy_Dict = {
                            "Algorithm" : "Kernel_Altenative_Proxy",
                            "Data_Size" : n_plus_m,
                            "Seed" : seed_,
                            "a_prime" : a_prime,
                            "sigma_square" : label_variance_in_eta_opt_,
                            "alternative_proxy_eta_parameter" : model.eta, 
                            "ATT_Estimation" : f_struct_pred,
                            "kernelW_length_scale": kernelW_length_scale_,
                            "Algo_Run_Time" : algo_run_time
                        }
    
                        df_results = pd.concat([df_results, pd.DataFrame([Kernel_Alternative_Proxy_Dict])], ignore_index=True)
    
                        df_results.to_pickle("../Results/SyntheticData_ATT_Estimation_KernelW_Bandwidth_Ablation.pkl")
