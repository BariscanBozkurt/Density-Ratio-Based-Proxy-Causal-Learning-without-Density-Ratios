import sys
sys.path.append("../../../src")

import os
from time import time
import numpy as np
import pandas as pd
import jax.numpy as jnp

from utils.kernel_utils import RBF
from causal_models.proxy_causal_learning import KernelAlternativeProxyATE
from utils.ml_utils import data_transform
from generate_experiment_data import read_deaner_dataset

if not os.path.exists("../../Results"):
    os.mkdir("../../Results")

data_path = '../../../data/deaner'
id_ = "IR"

data_seed_list = np.arange(100, 1100, 100)
seed_list = np.arange(0, 300, 100)
optimize_eta_list = [True]
stage1_perc_list = [0.5]
label_variance_in_eta_opt_list = [0.0, 1.0, 2.0, 3.0]

df_results = pd.DataFrame(columns = ["Algorithm", "Data_Size", "Seed", "Data_Seed", "Causal_MSE", "Causal_MAE", "eta_parameter", "sigma_square", "stage1_perc", "Algo_Run_Time"])

for data_seed_ in data_seed_list:
    for seed_ in seed_list:
        for optimize_eta in optimize_eta_list:
            for stage1_perc_ in stage1_perc_list:
                for label_variance_in_eta_opt_ in label_variance_in_eta_opt_list:
                    np.random.seed(seed_)

                    W, Z, A, Y, do_A, EY_do_A = read_deaner_dataset(data_path, id_, data_seed_)

                    W, Z, A, Y, do_A, EY_do_A = jnp.array(W), jnp.array(Z), jnp.array(A), jnp.array(Y), jnp.array(do_A), jnp.array(EY_do_A)

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
                    RBF_Kernel_W = RBF(use_length_scale_heuristic = True, use_jit_call = True)

                    lambda_ = 0.01
                    lambda2_ = 0.01
                    eta = 5e-3
                    optimize_lambda_parameters = True
                    optimize_eta_parameter = optimize_eta
                    lambda_optimization_range = (1e-7, 1.0)
                    eta_optimization_range = (1e-7, 1.0)
                    stage1_perc = stage1_perc_
                    regularization_grid_points = 150
                    make_psd_eps = 5*1e-9

                    model = KernelAlternativeProxyATE(
                                                        kernel_A = RBF_Kernel_A,
                                                        kernel_W = RBF_Kernel_W, 
                                                        kernel_Z = RBF_Kernel_Z,
                                                        lambda_ = lambda_,
                                                        eta = eta,
                                                        lambda2_ = lambda2_,
                                                        optimize_lambda_parameters = optimize_lambda_parameters,
                                                        optimize_eta_parameter = optimize_eta_parameter,
                                                        lambda_optimization_range = lambda_optimization_range,
                                                        eta_optimization_range = eta_optimization_range,
                                                        stage1_perc = stage1_perc,
                                                        regularization_grid_points = regularization_grid_points, 
                                                        make_psd_eps = make_psd_eps,
                                                        label_variance_in_eta_opt = label_variance_in_eta_opt_,
                                                    )
                    
                    model.fit((A_transformed, W_transformed, Z_transformed), Y_transformed)
                    do_A_size = do_A.shape[0]
                    do_A_transformed = (A_transformer.transform(do_A)).reshape(do_A_size, -1)
                    f_struct_pred_transformed = model.predict(do_A_transformed)
                    f_struct_pred = Y_transformer.inverse_transform(f_struct_pred_transformed.reshape(do_A_size, -1)).reshape(do_A_size, -1)
                    t1 = time()
                    algo_run_time = t1 - t0

                    structured_pred_mse = (jnp.mean((f_struct_pred.reshape(-1, 1) - EY_do_A.reshape(-1, 1)) ** 2)).item()
                    structured_pred_mae = (jnp.mean(np.abs(f_struct_pred.reshape(-1, 1) - EY_do_A.reshape(-1, 1)))).item()

                    if optimize_eta:
                        eta_parameter = "learned"
                    else:
                        eta_parameter = eta

                    Kernel_Alternative_Proxy_Dict = {
                        "Algorithm" : "Kernel_Altenative_Proxy",
                        "Data_Size" : W.shape[0],
                        "Seed" : seed_,
                        "Data_Seed" : data_seed_,
                        "Causal_MSE" : structured_pred_mse,
                        "Causal_MAE" : structured_pred_mae,
                        "eta_parameter" : eta_parameter,
                        "sigma_square" : label_variance_in_eta_opt_, 
                        "stage1_perc" : stage1_perc_,
                        "Algo_Run_Time" : algo_run_time
                    }

                    df_results = pd.concat([df_results, pd.DataFrame([Kernel_Alternative_Proxy_Dict])], ignore_index=True)

                    df_results.to_pickle("../../Results/Kernel_Alternative_Proxy_Deaner_IR.pkl")