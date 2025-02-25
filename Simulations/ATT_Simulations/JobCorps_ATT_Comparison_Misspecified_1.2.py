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
# from generate_experiment_data import generate_synthetic_data
from utils.ml_utils import data_transform
from generate_experiment_data import generate_train_jobcorp

if not os.path.exists("../Results"):
    os.mkdir("../Results")

def Lambda(t):
    return (0.9-0.1)*np.exp(t)/(1+np.exp(t)) + 0.1

seed_list = np.arange(0, 500, 100)
a_prime_list = [500, 1000]
label_variance_in_eta_opt_list = [0.0, 1.0]

scale_data = True
df_results = pd.DataFrame(columns = ["Algorithm", "Data_Size", "Seed", "a_prime", "sigma_square", "alternative_proxy_eta_parameter", "ATT_Estimation", "Algo_Run_Time"])

data_path = '../../data/JCdata.csv'
#######################################################
#        Alternative Kernel Proxy                     #
#######################################################

for seed_ in seed_list:
    for a_prime in a_prime_list:
        a_prime = jnp.array([a_prime])
        for label_variance_in_eta_opt_ in label_variance_in_eta_opt_list:
            np.random.seed(seed_)

            U, Y, A = generate_train_jobcorp(data_path)
            U = jnp.array(U, dtype = jnp.float64)
            Y = jnp.array(Y, dtype = jnp.float64)
            A = jnp.array(A, dtype = jnp.float64)

            Z = U + (2 * jnp.array(np.random.uniform(*U.shape)) - 1) * 1.
            W = Lambda(U / (U.max(1) + 1e-3).reshape(-1, 1)) + jnp.array(np.random.normal(*U.shape)) / 1.
            W = W[:,:20]

            do_A = jnp.linspace(40, 2000, 1000)[:, jnp.newaxis]

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
            RBF_Kernel_W = RBF(use_length_scale_heuristic = True, use_jit_call = True)
            RBF_Kernel_A = RBF(use_length_scale_heuristic = True, use_jit_call = True)

            lambda_ = 1e-2
            eta = 2*1e-3
            lambda2_ = 1e-3
            optimize_lambda_parameters = True
            optimize_eta_parameter = True
            optimize_zeta_parameter = True
            lambda_optimization_range = (5*1e-4, 1.0)
            lambda2_optimization_range = (5*1e-4, 1.0)
            zeta_optimization_range = (5*1e-4, 1.0)
            eta_optimization_range = (5*1e-4, 1.0)
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
                                                lambda2_optimization_range = lambda2_optimization_range,
                                                zeta_optimization_range = zeta_optimization_range,
                                                eta_optimization_range = eta_optimization_range,
                                                stage1_perc = stage1_perc,
                                                regularization_grid_points = regularization_grid_points, 
                                                label_variance_in_eta_opt = label_variance_in_eta_opt_,
                                                )
            do_A_size = do_A.shape[0]
            if scale_data:
                model.fit((A_transformed, W_transformed, Z_transformed), Y, A_transformer.transform(a_prime.reshape(-1, 1)))
                do_A_transformed = (A_transformer.transform(do_A)).reshape(do_A_size, -1)
                f_struct_pred = model.predict(do_A_transformed)
                # f_struct_pred = Y_transformer.inverse_transform(f_struct_pred_transformed.reshape(do_A_size, -1)).reshape(do_A_size, -1)
            else:
                model.fit((A, W, Z), Y, a_prime)
                f_struct_pred = model.predict(do_A).reshape(do_A_size, -1)

            f_struct_pred = np.array(f_struct_pred)
            t1 = time()
            algo_run_time = t1 - t0

            Kernel_Alternative_Proxy_Dict = {
                "Algorithm" : "Kernel_Alternative_Proxy",
                "Data_Size" : A.shape[0],
                "Seed" : seed_,
                "a_prime" : a_prime.item(),
                "sigma_square" : label_variance_in_eta_opt_,
                "alternative_proxy_eta_parameter" : model.eta, 
                "ATT_Estimation" : f_struct_pred,
                "Algo_Run_Time" : algo_run_time
            }

            df_results = pd.concat([df_results, pd.DataFrame([Kernel_Alternative_Proxy_Dict])], ignore_index=True)

            df_results.to_pickle("../Results/JobCorps_ATT_Estimation_Comparison_Misspecified_Setting1.2.pkl")

#######################################################
#        Kernel Negative Control                      #
#######################################################

for seed_ in seed_list:
    np.random.seed(seed_)

    U, Y, A = generate_train_jobcorp(data_path)
    U = jnp.array(U, dtype = jnp.float64)
    Y = jnp.array(Y, dtype = jnp.float64)
    A = jnp.array(A, dtype = jnp.float64)

    Z = U + (2 * jnp.array(np.random.uniform(*U.shape)) - 1) * 1.
    W = Lambda(U / (U.max(1) + 1e-3).reshape(-1, 1) ) + jnp.array(np.random.normal(*U.shape)) / 1.
    W = W[:,:20]

    do_A = jnp.linspace(40, 2000, 1000)[:, jnp.newaxis]

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
    RBF_Kernel_W = RBF(use_length_scale_heuristic = True, use_jit_call = True)
    RBF_Kernel_A = RBF(use_length_scale_heuristic = True, use_jit_call = True)

    lambda_ = 1e-3
    zeta = 1*1e-3
    lambda2 = 1e-3
    optimize_regularization_parameters = True
    lambda_optimization_range = (5*1e-4, 1.0)
    zeta_optimization_range = (5*1e-4, 1.0)
    stage1_perc = 0.5
    regularization_grid_points = 150

    model_KNC = KernelNegativeControlATT(
                                    kernel_A = RBF_Kernel_A,
                                    kernel_W = RBF_Kernel_W,
                                    kernel_Z = RBF_Kernel_Z,
                                    lambda_ = lambda_,
                                    zeta = zeta, 
                                    lambda2 = lambda2,
                                    optimize_regularization_parameters = optimize_regularization_parameters,
                                    lambda_optimization_range = lambda_optimization_range,
                                    zeta_optimization_range = zeta_optimization_range,
                                    )             
    do_A_size = do_A.shape[0]
    if scale_data:
        model_KNC.fit((A_transformed, W_transformed, Z_transformed), Y)
    else:
        model_KNC.fit((A, W, Z), Y) 
    t1 = time()
    algo_run_time = t1 - t0

    for a_prime in a_prime_list:
        a_prime = jnp.array([a_prime])
        if scale_data:
            do_A_transformed = (A_transformer.transform(do_A)).reshape(do_A_size, -1)
            f_struct_pred_knc = model_KNC.predict(do_A_transformed, A_transformer.transform(a_prime.reshape(-1, 1)))
            # f_struct_pred_knc = np.array(Y_transformer.inverse_transform(f_struct_pred_transformed.reshape(do_A_size, -1)).reshape(do_A_size, -1))
        else:
            f_struct_pred_knc = np.array(model_KNC.predict(do_A, a_prime).reshape(do_A_size, -1))

        Kernel_Negative_Control_Dict = {
            "Algorithm" : "Kernel_Negative_Control",
            "Data_Size" : A.shape[0],
            "Seed" : seed_,
            "a_prime" : a_prime.item(),
            "sigma_square" : None,
            "alternative_proxy_eta_parameter" : None,
            "ATT_Estimation" : f_struct_pred_knc,
            "Algo_Run_Time" : algo_run_time
        }

        df_results = pd.concat([df_results, pd.DataFrame([Kernel_Negative_Control_Dict])], ignore_index=True)

        df_results.to_pickle("../Results/JobCorps_ATT_Estimation_Comparison_Misspecified_Setting1.2.pkl")

#######################################################
#        Kernel ATT                                   #
#######################################################
for seed_ in seed_list:
    np.random.seed(seed_)
    U, Y, A = generate_train_jobcorp(data_path)
    U = jnp.array(U, dtype = jnp.float64)
    Y = jnp.array(Y, dtype = jnp.float64)
    A = jnp.array(A, dtype = jnp.float64)

    Z = U + (2 * jnp.array(np.random.uniform(*U.shape)) - 1) * 1.
    W = Lambda(U / (U.max(1) + 1e-3).reshape(-1, 1) ) + jnp.array(np.random.normal(*U.shape)) / 1.
    W = W[:,:20]

    do_A = jnp.linspace(40, 2000, 1000)[:, jnp.newaxis]

    if scale_data:
        A_transformed, A_transformer = data_transform(A)
        U_transformed, U_transformer = data_transform(U)
        Y_transformed, Y_transformer = data_transform(Y)

        data_size = A_transformed.shape[0]
        A_transformed = jnp.array(A_transformed).reshape(data_size, -1)
        U_transformed = jnp.array(U_transformed).reshape(data_size, -1)
        Y_transformed = jnp.array(Y_transformed).reshape(data_size, -1)

    t0 = time()
    RBF_Kernel_A_ = RBF(use_length_scale_heuristic = True, use_jit_call = True)
    RBF_Kernel_X_ = RBF(use_length_scale_heuristic = True, use_jit_call = True)

    model_KernelATT = KernelATT(
                                kernel_A = RBF_Kernel_A_,
                                kernel_X = RBF_Kernel_X_,
                                optimize_regularization_parameters = True,
                                lambda_optimization_range = (1e-6, 1.0),
                                regularization_grid_points = 150,
                                )
    do_A_size = do_A.shape[0]
    if scale_data:
        model_KernelATT.fit((A_transformed, U_transformed), Y)
    else:
        model_KernelATT.fit((A, U), Y)
    t1 = time()
    algo_run_time = t1 - t0

    for a_prime in a_prime_list:
        a_prime = jnp.array([a_prime])
        if scale_data:
            do_A_transformed = (A_transformer.transform(do_A)).reshape(do_A_size, -1)
            f_struct_pred_katt = model_KernelATT.predict(do_A_transformed, A_transformer.transform(a_prime.reshape(-1, 1)))
            # f_struct_pred_katt = np.array(Y_transformer.inverse_transform(f_struct_pred_transformed.reshape(do_A_size, -1)).reshape(do_A_size, -1))
        else:
            f_struct_pred_katt = np.array(model_KernelATT.predict(do_A, a_prime).reshape(do_A_size, -1))

        Kernel_ATT_Dict = {
            "Algorithm" : "Kernel_ATT",
            "Data_Size" : A.shape[0],
            "Seed" : seed_,
            "a_prime" : a_prime.item(),
            "sigma_square" : None,
            "alternative_proxy_eta_parameter" : None,
            "ATT_Estimation" : f_struct_pred_katt,
            "Algo_Run_Time" : algo_run_time
        }

        df_results = pd.concat([df_results, pd.DataFrame([Kernel_ATT_Dict])], ignore_index=True)

        df_results.to_pickle("../Results/JobCorps_ATT_Estimation_Comparison_Misspecified_Setting1.2.pkl")