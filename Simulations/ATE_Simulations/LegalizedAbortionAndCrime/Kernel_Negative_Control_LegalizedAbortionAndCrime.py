import sys
sys.path.append("../../../src")

import os
from time import time
import numpy as np
import pandas as pd
import jax.numpy as jnp

from utils.kernel_utils import RBF
from causal_models.proxy_causal_learning import KernelNegativeControlATE
from generate_experiment_data import read_legalized_abortion_and_crime_dataset
from utils.ml_utils import data_transform

if not os.path.exists("../../Results"):
    os.mkdir("../../Results")

data_path = '../../../data/abortion'

data_seed_list = np.arange(0, 10)
seed_list = np.arange(0, 300, 100)
scale_data_list = [True, False]

df_results = pd.DataFrame(columns = ["Algorithm", "Data_Size", "Seed", "Data_Seed", "Causal_MSE", "Causal_MAE", "scale_data", "Algo_Run_Time"])

for data_seed_ in data_seed_list:
    for seed_ in seed_list:
        for scale_data in scale_data_list:
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
            RBF_Kernel_Z = RBF(use_length_scale_heuristic = True, use_jit_call = True)
            RBF_Kernel_W = RBF(use_length_scale_heuristic = True, use_jit_call = True)
            RBF_Kernel_A = RBF(use_length_scale_heuristic = True, use_jit_call = True)
            # RBF_Kernel_X = RBF(use_length_scale_heuristic = True, use_jit_call = True)

            lambda_ = 0.1
            zeta = 5*1e-4
            optimize_regularization_parameters = True
            lambda_optimization_range = (1e-15, 1.0)
            stage1_perc = 0.5
            regularization_grid_points = 150

            model = KernelNegativeControlATE(
                                            kernel_A = RBF_Kernel_A,
                                            kernel_W = RBF_Kernel_W,
                                            kernel_Z = RBF_Kernel_Z,
                                            lambda_ = lambda_,
                                            zeta = zeta, 
                                            optimize_regularization_parameters = True,
                                            lambda_optimization_range = (1e-9, 1.0),
                                            zeta_optimization_range = (1e-9, 1.0),
                                            large_lambda_zeta_option = False,
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

            Kernel_Negative_Control_Dict = {
                "Algorithm" : "Kernel_Negative_Control",
                "Data_Size" : W.shape[0],
                "Seed" : seed_,
                "Data_Seed" : data_seed_,
                "Causal_MSE" : structured_pred_mse,
                "Causal_MAE" : structured_pred_mae,
                "scale_data" : scale_data,
                "Algo_Run_Time" : algo_run_time
            }

            df_results = pd.concat([df_results, pd.DataFrame([Kernel_Negative_Control_Dict])], ignore_index=True)

            df_results.to_pickle("../../Results/Kernel_Negative_Control_Abortion_and_Crime.pkl")