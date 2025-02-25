import sys
sys.path.append("../../../src/other_methods")
sys.path.append("../../..")

import os
from time import time
import numpy as np
import pandas as pd

from PMMR.kpv import KernelPVModelV2
from PMMR.data_class import PVTrainDataSet, PVTestDataSet
from src.generate_experiment_data import generate_synthetic_data
from src.utils.ml_utils import data_transform

if not os.path.exists("../../Results"):
    os.mkdir("../../Results")

data_size_list = [500, 1000, 2000]
seed_list = np.arange(0, 3000, 100)
scale_data_list = [True, False]

df_results = pd.DataFrame(columns = ["Algorithm", "Data_Size", "Seed", "Causal_MSE", "Causal_MAE", "scale_data", "Algo_Run_Time"])

for n_plus_m in data_size_list:
    for seed_ in seed_list:
        for scale_data in scale_data_list:
            np.random.seed(seed_)
            
            U, W, Z, A, Y, do_A, EY_do_A = generate_synthetic_data(size = n_plus_m, seed = seed_)

            if scale_data:
                A_transformed, A_transformer = data_transform(A)
                Z_transformed, Z_transformer = data_transform(Z)
                W_transformed, W_transformer = data_transform(W)
                Y_transformed, Y_transformer = data_transform(Y)

                data_size = A_transformed.shape[0]
                A_transformed = np.array(A_transformed).reshape(data_size, -1)
                Z_transformed = np.array(Z_transformed).reshape(data_size, -1)
                W_transformed = np.array(W_transformed).reshape(data_size, -1)
                Y_transformed = np.array(Y_transformed).reshape(data_size, -1)

                train_dataset = PVTrainDataSet( treatment = A_transformed,
                                                treatment_proxy = Z_transformed,
                                                outcome_proxy = W_transformed,
                                                outcome = Y_transformed,
                                                backdoor = None)
            else:
                train_dataset = PVTrainDataSet( treatment = A,
                                                treatment_proxy = Z,
                                                outcome_proxy = W,
                                                outcome = Y,
                                                backdoor = None)

            test_dataset = PVTestDataSet(treatment = do_A,
                                        structural = EY_do_A)

            t0 = time()
            kpv_params = {
                    "lam1_max": 0.1,
                    "lam1_min": 0.001,
                    "n_lam1_search": 10,
                    "lam2": 0.001,
                    "split_ratio": 0.5,
                    "scale": 0.5,
                }


            model = KernelPVModelV2(**kpv_params)

            model.fit(train_dataset, "synthetic_low_dim")

            if scale_data:
                do_A_size = do_A.shape[0]
                do_A_transformed = (A_transformer.transform(do_A.reshape(-1, 1))).reshape(do_A_size, -1)
                f_struct_pred_transformed = model.predict(do_A_transformed)
                f_struct_pred = Y_transformer.inverse_transform(f_struct_pred_transformed.reshape(do_A_size, -1)).reshape(do_A_size, -1)
            else:
                f_struct_pred = model.predict(do_A)
            t1 = time()
            algo_run_time = t1 - t0

            structured_pred_mse = (np.mean((f_struct_pred.reshape(-1, 1) - EY_do_A.reshape(-1, 1)) ** 2)).item()
            structured_pred_mae = (np.mean(np.abs(f_struct_pred.reshape(-1, 1) - EY_do_A.reshape(-1, 1)))).item()

            Kernel_Alternative_Proxy_Dict = {
                "Algorithm" : "KPV",
                "Data_Size" : n_plus_m,
                "Seed" : seed_,
                "Causal_MSE" : structured_pred_mse,
                "Causal_MAE" : structured_pred_mae,
                "scale_data" : scale_data,
                "Algo_Run_Time" : algo_run_time
            }

            df_results = pd.concat([df_results, pd.DataFrame([Kernel_Alternative_Proxy_Dict])], ignore_index=True)

            df_results.to_pickle("../../Results/KPV_Synthetic_Experiment1_V2.pkl")