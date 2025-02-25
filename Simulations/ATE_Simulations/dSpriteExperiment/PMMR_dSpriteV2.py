import sys
sys.path.append("../../../src/other_methods")
sys.path.append("../../..")

import os
from time import time
import numpy as np
import pandas as pd

from PMMR.pmmr import PMMRModelV2
from PMMR.data_class import PVTrainDataSet, PVTestDataSet
from src.dsprite_ver2 import *
from src.utils.ml_utils import data_transform

if not os.path.exists("../../Results"):
    os.mkdir("../../Results")

data_path = '../../../data/dsprite'
data_size_list = [500, 1000, 2000]
seed_list = np.arange(0, 3000, 100)
lam1_list = [1e-4, 1e-3, 1e-2]
lam2_list = [1e-4, 1e-3, 1e-2]
scale_data_list = [False, True]

df_results = pd.DataFrame(columns = ["Algorithm", "Data_Size", "Seed", "lam1", "lam2", "Causal_MSE", "Causal_MAE", "scale_data", "Algo_Run_Time"])

for n_plus_m in data_size_list:
    for seed_ in seed_list:
        for lam1_ in lam1_list:
            for lam2_ in lam2_list:
                for scale_data in scale_data_list:
                    np.random.seed(seed_)
                    
                    train_dataset = generate_train_dsprite_ver2(n_sample = n_plus_m, rand_seed = seed_)
                    test_dataset = generate_test_dsprite_ver2()
                    do_A = test_dataset.treatment
                    EY_do_A = test_dataset.structural

                    if scale_data:
                        A = train_dataset.treatment
                        Y = train_dataset.outcome
                        Z = train_dataset.treatment_proxy
                        W = train_dataset.outcome_proxy

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
                    t0 = time()
                    model = PMMRModelV2(lam1 = lam1_, lam2 = lam2_, scale = 0.5)

                    if scale_data:
                        model.fit(train_dataset, "dSprite")
                    else:
                        model.fit(train_dataset, "dsprite")

                    if scale_data:
                        do_A_size = do_A.shape[0]
                        do_A_transformed = (A_transformer.transform(do_A))
                        f_struct_pred_transformed = model.predict(do_A_transformed)
                        f_struct_pred = Y_transformer.inverse_transform(f_struct_pred_transformed.reshape(do_A_size, -1)).reshape(do_A_size, -1)
                    else:
                        f_struct_pred = model.predict(do_A)
                    t1 = time()
                    algo_run_time = t1 - t0

                    structured_pred_mse = (np.mean((f_struct_pred.reshape(-1, 1) - EY_do_A.reshape(-1, 1)) ** 2)).item()
                    structured_pred_mae = (np.mean(np.abs(f_struct_pred.reshape(-1, 1) - EY_do_A.reshape(-1, 1)))).item()

                    Kernel_Alternative_Proxy_Dict = {
                        "Algorithm" : "PMMR",
                        "Data_Size" : n_plus_m,
                        "Seed" : seed_,
                        "lam1": lam1_,
                        "lam2": lam2_,
                        "Causal_MSE" : structured_pred_mse,
                        "Causal_MAE" : structured_pred_mae,
                        "scale_data" : scale_data,
                        "Algo_Run_Time" : algo_run_time
                    }

                    df_results = pd.concat([df_results, pd.DataFrame([Kernel_Alternative_Proxy_Dict])], ignore_index=True)

                    df_results.to_pickle("../../Results/PMMR_dSpriteV2.pkl")