import sys
sys.path.append("../../../src")

import os
from time import time
import numpy as np
import pandas as pd
import jax.numpy as jnp

from utils.kernel_utils import RBF, MaternKernel
from causal_models.proxy_causal_learning import KernelAlternativeProxyATE
from generate_experiment_data import read_legalized_abortion_and_crime_dataset
from utils.ml_utils import data_transform

if not os.path.exists("../../Results"):
    os.mkdir("../../Results")

data_path = '../../../data/abortion'

data_seed_list = np.arange(0, 10)
seed_list = np.arange(0, 300, 100)
optimize_eta_list = [True]
scale_data_list = [True]
kernel_list = ["RBF"] #["RBF", "Matern"]
stage1_perc_list = [0.5]
label_variance_in_eta_opt_list = [0.0, 1.0, 2.0, 3.0]

df_results = pd.DataFrame(columns = ["Algorithm", "Data_Size", "Seed", "Data_Seed", "Causal_MSE", "Causal_MAE", "eta_parameter", "sigma_square", "scale_data", "kernel", "stage1_perc", "Algo_Run_Time"])

for data_seed_ in data_seed_list:
    for seed_ in seed_list:
        for optimize_eta in optimize_eta_list:
            for scale_data in scale_data_list:
                for kernel_ in kernel_list:
                    for stage1_perc_ in stage1_perc_list:
                        for label_variance_in_eta_opt_ in label_variance_in_eta_opt_list:
                            np.random.seed(seed_)

                            W, Z, A, Y, do_A, EY_do_A = read_legalized_abortion_and_crime_dataset(data_path, seed = data_seed_)
                            
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

                            else:
                                W, Z, A, Y, do_A, EY_do_A = jnp.array(W), jnp.array(Z), jnp.array(A), jnp.array(Y), jnp.array(do_A), jnp.array(EY_do_A)

                            t0 = time()
                            if kernel_ == "RBF":
                                RBF_Kernel_Z = RBF(use_length_scale_heuristic = True, use_jit_call = True)
                                RBF_Kernel_A = RBF(use_length_scale_heuristic = True, use_jit_call = True)
                                RBF_Kernel_W = RBF(use_length_scale_heuristic = True, use_jit_call = True)
                            elif kernel_ == "Matern":
                                p = 2
                                RBF_Kernel_Z = MaternKernel(p, use_length_scale_heuristic = True)
                                RBF_Kernel_A = MaternKernel(p, use_length_scale_heuristic = True)
                                RBF_Kernel_W = MaternKernel(p, use_length_scale_heuristic = True)

                            if scale_data:
                                eta = 1e-3
                            else:
                                eta = 5*1e-3
                            lambda_ = 0.01
                            eta = 1*1e-3
                            lambda2_ = 0.01
                            optimize_lambda_parameters = True
                            optimize_eta_parameter = optimize_eta
                            lambda_optimization_range = (1e-7, 1.0)
                            eta_optimization_range = (1e-7, 1.0)
                            stage1_perc = stage1_perc_
                            regularization_grid_points = 150
                            make_psd_eps = 5*1e-9

                            model = KernelAlternativeProxyATE(  kernel_A = RBF_Kernel_A,
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
                            
                            if scale_data:
                                model.fit((A_transformed, W_transformed, Z_transformed), Y_transformed)
                                do_A_size = do_A.shape[0]
                                do_A_transformed = (A_transformer.transform(do_A)).reshape(do_A_size, -1)
                                f_struct_pred_transformed = model.predict(do_A_transformed)
                                f_struct_pred = Y_transformer.inverse_transform(f_struct_pred_transformed.reshape(do_A_size, -1)).reshape(do_A_size, -1)
                            else:
                                model.fit((A, W, Z), Y)
                                f_struct_pred = model.predict(do_A)

                            t1 = time()
                            algo_run_time = t1 - t0

                            structured_pred_mse = (jnp.mean((f_struct_pred.reshape(-1, 1) - EY_do_A.reshape(-1, 1)) ** 2)).item()
                            structured_pred_mae = (jnp.mean(np.abs(f_struct_pred.reshape(-1, 1) - EY_do_A.reshape(-1, 1)))).item()

                            if optimize_eta:
                                eta_parameter = "learned"
                            else:
                                eta_parameter = eta

                            Kernel_Alternative_Proxy_Dict = {
                                "Algorithm" : "Kernel_Alternative_Proxy",
                                "Data_Size" : W.shape[0],
                                "Seed" : seed_,
                                "Data_Seed" : data_seed_,
                                "Causal_MSE" : structured_pred_mse,
                                "Causal_MAE" : structured_pred_mae,
                                "eta_parameter" : eta_parameter,
                                "sigma_square" : label_variance_in_eta_opt_,
                                "scale_data" : scale_data,
                                "kernel" : kernel_,
                                "stage1_perc" : stage1_perc_,
                                "Algo_Run_Time" : algo_run_time
                            }

                            df_results = pd.concat([df_results, pd.DataFrame([Kernel_Alternative_Proxy_Dict])], ignore_index=True)

                            df_results.to_pickle("../../Results/Kernel_Alternative_Proxy_Abortion_and_Crime.pkl")