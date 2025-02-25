import sys
sys.path.append("../../../src/other_methods/DoublyRobustPCL")
sys.path.append("../../..")

import yaml
import os
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.data_class import PVTrainDataSet,PVTestDataSet
from model.rkhs.Trainer import RKHS_Trainer
from src.generate_experiment_data import read_legalized_abortion_and_crime_dataset
from src.utils.ml_utils import data_transform

cfg = "../../../src/other_methods/DoublyRobustPCL/config/job/train/default.yaml"

with open(cfg) as stream:
    try:
        cfg = yaml.safe_load(stream)
        # print(yaml.safe_load(stream))
    except yaml.YAMLError as exc:
        print(exc)

class obj(object):
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(k, (list, tuple)):
                setattr(self, k, [obj(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, obj(v) if isinstance(v, dict) else v)

cfg = obj(cfg)

if not os.path.exists("../../Results"):
    os.mkdir("../../Results")

data_path = '../../../data/abortion'

data_seed_list = np.arange(0, 10)
seed_list = np.arange(0, 300, 100)
scale_data_list = [True, False]

df_results = pd.DataFrame(columns = ["Algorithm", "Data_Size", "Seed", "Causal_MSE", "Causal_MAE", "scale_data", "Algo_Run_Time"])

for data_seed_ in data_seed_list:
    for seed_ in seed_list:
        for scale_data in scale_data_list:
            np.random.seed(seed_)
            return_test = True
            W, Z, A, Y, W_test, Z_test, A_test, Y_test, do_A, EY_do_A = read_legalized_abortion_and_crime_dataset(data_path, return_test, data_seed_)

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

                test_shape = A_test.shape[0]
                A_test_transformed = A_transformer.transform(A_test).reshape(test_shape, -1)
                Z_test_transformed = Z_transformer.transform(Z_test).reshape(test_shape, -1)
                W_test_transformed = W_transformer.transform(W_test).reshape(test_shape, -1)
                Y_test_transformed = Y_transformer.transform(Y_test).reshape(test_shape, -1)

                do_A_size = do_A.shape[0]
                do_A_transformed = (A_transformer.transform(do_A)).reshape(do_A_size, -1)
                EY_do_A_transformed = (Y_transformer.transform(EY_do_A)).reshape(do_A_size, -1)

                train_dataset = PVTrainDataSet( treatment = A_transformed,
                                                treatment_proxy = Z_transformed,
                                                outcome_proxy = W_transformed,
                                                outcome = Y_transformed,
                                                backdoor = None)


                test_dataset = PVTrainDataSet(  treatment = A_test_transformed,
                                                treatment_proxy = Z_test_transformed,
                                                outcome_proxy = W_test_transformed,
                                                outcome = Y_test_transformed,
                                                backdoor = None)
            else:
                train_dataset = PVTrainDataSet( treatment = A,
                                                treatment_proxy = Z,
                                                outcome_proxy = W,
                                                outcome = Y,
                                                backdoor = None)

                test_dataset = PVTrainDataSet(  treatment = A_test,
                                                treatment_proxy = Z_test,
                                                outcome_proxy = W_test,
                                                outcome = Y_test,
                                                backdoor = None)
            
            t0 = time()
            rkhs_train = RKHS_Trainer(cfg, train_dataset)
            if scale_data:
                # Train q
                rkhs_train.fit_q_cv()
                f_struct_pred_transformed = np.array(rkhs_train._qtest(do_A_transformed, test_dataset))

                f_struct_pred = Y_transformer.inverse_transform(f_struct_pred_transformed.reshape(do_A_size, -1)).reshape(do_A_size, -1)
            else:
                # Train q
                rkhs_train.fit_q_cv()
                ATE_q = rkhs_train._qtest(do_A, test_dataset)
                f_struct_pred = np.array(ATE_q)
            t1 = time()

            algo_run_time = t1 - t0

            structured_pred_mse = (np.mean((f_struct_pred.reshape(-1, 1) - EY_do_A.reshape(-1, 1)) ** 2)).item()
            structured_pred_mae = (np.mean(np.abs(f_struct_pred.reshape(-1, 1) - EY_do_A.reshape(-1, 1)))).item()

            PKIPW_dict = {
                "Algorithm" : "PKIPW",
                "Data_Size" : W.shape[0],
                "Seed" : seed_,
                "Data_Seed" : data_seed_,
                "Causal_MSE" : structured_pred_mse,
                "Causal_MAE" : structured_pred_mae,
                "scale_data" : scale_data,
                "Algo_Run_Time" : algo_run_time
            }

            df_results = pd.concat([df_results, pd.DataFrame([PKIPW_dict])], ignore_index=True)

            df_results.to_pickle("../../Results/PKIPW_Abortion_and_Crime.pkl")