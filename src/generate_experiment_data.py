import os
import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy.sparse import diags
from scipy.stats import norm
from itertools import product

def generate_train_jobcorp(data_path = "data/JCdata.csv"):
    # taken from https://github.com/liyuan9988/KernelCausalFunction/blob/master/src/ate/data/job_corp.py
    data = pd.read_csv(data_path, sep=" ")

    sub = data
    sub = data.loc[data["m"] > 0, :]
    sub = sub.loc[sub["d"] >= 40, :]
    outcome = sub["m"].to_numpy()
    treatment = sub["d"].to_numpy()
    backdoor = sub.iloc[:, 3:].to_numpy()
    return backdoor, outcome[:,np.newaxis], treatment[:,np.newaxis]

def generate_synthetic_data(size = 1000, beta = 1, sigma = 1, do_A_range = (-1, 2), do_A_size = 100, seed = 10):
    """
    This data generation process is taken from Appendix H of the following paper:
    Doubly Robust Proximal Causal Learning for Continuous Treatments, Yong Wu, Yanwei Fu, Shouyan Wang, Xinwei Sun 
    """
    np.random.seed(seed)
    
    U2 = np.random.uniform(-1, 2, size = size)
    U1 = np.random.uniform(0, 1, size = size) - ((U2 >= 0) & (U2 <= 1)).astype(int)
    U = np.c_[U1, U2]
    
    Z2 = U2 + np.random.uniform(-1, 1, size = size)
    Z1 = U1 + np.random.normal(0, sigma, size = size)
    Z = np.c_[Z1, Z2]

    W1 = U1 + np.random.uniform(-1, 1, size = size)
    W2 = U2 + np.random.normal(0, sigma, size = size)
    W = np.c_[W1, W2]
    
    A = U2 + np.random.normal(0, beta, size = size)
    Y = 3 * np.cos(2 * (0.3 * U1 + 0.3 * U2 + 0.2) + 1.5 * A) + np.random.normal(0, 1, size = size)

    A, Y = A.reshape(size, -1), Y.reshape(size, -1)
    do_A = np.linspace(do_A_range[0], do_A_range[1], do_A_size)

    EY_do_A = []
    for a_ in do_A:
        U2 = np.random.uniform(-1, 2, size = 10000)
        U1 = np.random.uniform(0, 1, size = 10000) - ((U2 >= 0) & (U2 <= 1)).astype(int)
        EY_do_A.append(np.mean(3 * np.cos(2 * ( 0.3 * U1 + 0.3 * U2 + 0.2) + 1.5 * a_)))
    EY_do_A = np.array(EY_do_A)

    do_A, EY_do_A = do_A.reshape(do_A_size, -1), EY_do_A.reshape(do_A_size, -1)
    return U, W, Z, A, Y, do_A, EY_do_A


def Lambda(t):
    return (0.9-0.1)*np.exp(t)/(1+np.exp(t)) + 0.1


def read_legalized_abortion_and_crime_dataset(data_path: str,
                                              return_test: bool = False,
                                              seed: int = 0):
    seed_str = str(seed)
    folder_path_train = data_path + '/train'
    folder_path_effect = data_path + '/test'
    
    train_name = f'main_ab_seed{seed_str}.npz'
    train_path = f'{folder_path_train}/{train_name}'
    train_data = np.load(train_path)
    W, Z, A, Y = train_data['train_w'], train_data['train_z'], train_data['train_a'], train_data['train_y']

    effect_name = f'do_A_ab_seed{seed_str}.npz'
    effect_path = f'{folder_path_effect}/{effect_name}'
    effect_data = np.load(effect_path)
    do_A, EY_do_A = effect_data['do_A'], effect_data['gt_EY_do_A']
    if return_test:
        W_test, Z_test, A_test, Y_test = train_data['test_w'], train_data['test_z'], train_data['test_a'], train_data['test_y']
        return W, Z, A, Y, W_test, Z_test, A_test, Y_test, do_A, EY_do_A
    else:
        return W, Z, A, Y, do_A, EY_do_A


def read_deaner_dataset(data_path: str,
                        id_: str,
                        seed: int,
                        return_test: bool = False):
    id_path = id_ + "_80_N"
    data_path = os.path.join(data_path, id_path)
    npz_train_file = f"main_edu_{id_}_80_seed{seed}.npz"
    npz_effect_file = f"do_A_edu_{id_}_80_seed{seed}.npz"
    train_data = np.load(os.path.join(data_path, npz_train_file))
    effect_data = np.load(os.path.join(data_path, npz_effect_file))
    W, Z, A, Y = train_data['train_w'], train_data['train_z'], train_data['train_a'], train_data['train_y']
    do_A, EY_do_A = effect_data['do_A'], effect_data['gt_EY_do_A'].reshape(-1, 1)
    if return_test:
        W_test, Z_test, A_test, Y_test = train_data['test_w'], train_data['test_z'], train_data['test_a'], train_data['test_y']
        return W, Z, A, Y, W_test, Z_test, A_test, Y_test, do_A, EY_do_A
    else:
        return W, Z, A, Y, do_A, EY_do_A


class dSprite_ProxyVariable_Dataset():

    def __init__(self,):
        pass
        
    def image_id(self, latent_bases: np.ndarray, posX_id_arr: np.ndarray, posY_id_arr: np.ndarray,
                 orientation_id_arr: np.ndarray,
                 scale_id_arr: np.ndarray):
        data_size = posX_id_arr.shape[0]
        color_id_arr = np.array([0] * data_size, dtype=int)
        shape_id_arr = np.array([2] * data_size, dtype=int)
        idx = np.c_[color_id_arr, shape_id_arr, scale_id_arr, orientation_id_arr, posX_id_arr, posY_id_arr]
        return idx.dot(latent_bases)
    
    
    def structural_func(self, image, weights):
        return (np.mean((image.dot(weights)) ** 2, axis=1) - 5000) / 1000
    
    
    def generate_test_dsprite(self, data_path: str):
        dataset_zip = np.load(os.path.join(data_path, "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"),
                              allow_pickle=True, encoding="bytes")
        weights = np.load(os.path.join(data_path, "dsprite_mat.npy"))
    
        imgs = dataset_zip['imgs']
        latents_values = dataset_zip['latents_values']
        metadata = dataset_zip['metadata'][()]
    
        latents_sizes = metadata[b'latents_sizes']
        latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                        np.array([1, ])))
    
        posX_id_arr = [0, 5, 10, 15, 20, 25, 30]
        posY_id_arr = [0, 5, 10, 15, 20, 25, 30]
        scale_id_arr = [0, 3, 5]
        orientation_arr = [0, 10, 20, 30]
        latent_idx_arr = []
        for posX, posY, scale, orientation in product(posX_id_arr, posY_id_arr, scale_id_arr, orientation_arr):
            latent_idx_arr.append([0, 2, scale, orientation, posX, posY])
    
        image_idx_arr = np.array(latent_idx_arr).dot(latents_bases)
        data_size = 7 * 7 * 3 * 4
        treatment = imgs[image_idx_arr].reshape((data_size, 64 * 64))
        structural = self.structural_func(treatment, weights)
        structural = structural[:, np.newaxis]
        return treatment, structural
    
    
    def generate_dsprite_pv(self, data_path: str, 
                            n_sample: int,
                            rand_seed: int = 42, **kwargs):
        """
        Parameters
        ----------
        n_sample : int
            size of data
        rand_seed : int
            random seed
    
        Returns
        -------
        train_data : TrainDataSet
        """
        dataset_zip = np.load(os.path.join(data_path, "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"),
                              allow_pickle=True, encoding="bytes")
        weights = np.load(os.path.join(data_path, "dsprite_mat.npy"))
    
        imgs = dataset_zip['imgs']
        latents_values = dataset_zip['latents_values']
        metadata = dataset_zip['metadata'][()]
    
        latents_sizes = metadata[b'latents_sizes']
        latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                        np.array([1, ])))
    
        rng = default_rng(seed=rand_seed)
        posX_id_arr = rng.integers(32, size=n_sample)
        posY_id_arr = rng.integers(32, size=n_sample)
        scale_id_arr = rng.integers(6, size=n_sample)
        orientation_arr = rng.integers(40, size=n_sample)
        image_idx_arr = self.image_id(latents_bases, posX_id_arr, posY_id_arr, orientation_arr, scale_id_arr)
        treatment = imgs[image_idx_arr].reshape((n_sample, 64 * 64)).astype(np.float64)
        treatment += rng.normal(0.0, 0.1, treatment.shape)
        latent_feature = latents_values[image_idx_arr]  # (color, shape, scale, orientation, posX, posY)
        treatment_proxy = latent_feature[:, 2:5]  # (scale, orientation, posX)
    
        posX_id_proxy = np.array([16] * n_sample)
        scale_id_proxy = np.array([3] * n_sample)
        orientation_proxy = np.array([0] * n_sample)
        proxy_image_idx_arr = self.image_id(latents_bases, posX_id_proxy, posY_id_arr, orientation_proxy, scale_id_proxy)
        outcome_proxy = imgs[proxy_image_idx_arr].reshape((n_sample, 64 * 64)).astype(np.float64)
        outcome_proxy += rng.normal(0.0, 0.1, outcome_proxy.shape)
    
        structural = self.structural_func(treatment, weights)
        outcome = structural * (posY_id_arr - 15.5) ** 2 / 85.25 + rng.normal(0.0, 0.5, n_sample)
        outcome = outcome[:, np.newaxis]
        
        do_A, EY_do_A = self.generate_test_dsprite(data_path)
        return treatment, outcome, treatment_proxy, outcome_proxy, do_A, EY_do_A


class dSprite_ProxyVariable_DatasetV2():
    ### This is based on "Update on dSprite experiment", see https://github.com/liyuan9988/DeepFeatureProxyVariable/tree/master
    def __init__(self,):
        pass
        
    def cal_weight(self,):
        weights = np.empty((64, 64))
        for i in range(64):
            for j in range(64):
                weights[i, j] = (np.abs(32 - j))
        return weights.reshape(64*64, 1) / 32


    def image_id(self, latent_bases: np.ndarray, posX_id_arr: np.ndarray, posY_id_arr: np.ndarray,
                 orientation_id_arr: np.ndarray,
                 scale_id_arr: np.ndarray):
        data_size = posX_id_arr.shape[0]
        color_id_arr = np.array([0] * data_size, dtype=int)
        shape_id_arr = np.array([2] * data_size, dtype=int)
        idx = np.c_[color_id_arr, shape_id_arr, scale_id_arr, orientation_id_arr, posX_id_arr, posY_id_arr]
        return idx.dot(latent_bases)
        
    
    def structural_func(self, image, weights):
        # return (image.dot(weights)[:, 0] ** 2 - 5000) / 1000
        return (image.dot(weights)[:, 0] ** 2 - 3000) / 500
    
    
    def generate_test_dsprite(self, data_path: str):
        dataset_zip = np.load(os.path.join(data_path, "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"),
                              allow_pickle=True, encoding="bytes")
        
        weights = self.cal_weight()
    
        imgs = dataset_zip['imgs']
        latents_values = dataset_zip['latents_values']
        metadata = dataset_zip['metadata'][()]

        latents_sizes = metadata[b'latents_sizes']
        latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                        np.array([1, ])))

        posX_id_arr = [0, 5, 10, 15, 20, 25, 30]
        posY_id_arr = [0, 5, 10, 15, 20, 25, 30]
        scale_id_arr = [0, 3, 5]
        orientation_arr = [0, 10, 20, 30]
        latent_idx_arr = []
        for posX, posY, scale, orientation in product(posX_id_arr, posY_id_arr, scale_id_arr, orientation_arr):
            latent_idx_arr.append([0, 2, scale, orientation, posX, posY])

        image_idx_arr = np.array(latent_idx_arr).dot(latents_bases)
        data_size = 7 * 7 * 3 * 4
        treatment = imgs[image_idx_arr].reshape((data_size, 64 * 64))
        structural = self.structural_func(treatment, weights)
        structural = structural[:, np.newaxis]
        return treatment, structural
    
    
    def generate_dsprite_pv(self, data_path: str, 
                            n_sample: int,
                            rand_seed: int = 42, **kwargs):
        """
        Parameters
        ----------
        n_sample : int
            size of data
        rand_seed : int
            random seed
    
        Returns
        -------
        train_data : TrainDataSet
        """
        dataset_zip = np.load(os.path.join(data_path, "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"),
                              allow_pickle=True, encoding="bytes")
        weights = np.load(os.path.join(data_path, "dsprite_mat.npy"))
    
        imgs = dataset_zip['imgs']
        latents_values = dataset_zip['latents_values']
        metadata = dataset_zip['metadata'][()]
    
        latents_sizes = metadata[b'latents_sizes']
        latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                        np.array([1, ])))
    
        rng = default_rng(seed=rand_seed)
        posX_id_arr = rng.integers(32, size=n_sample)
        posY_id_arr = rng.integers(32, size=n_sample)
        scale_id_arr = rng.integers(6, size=n_sample)
        orientation_arr = rng.integers(40, size=n_sample)
        image_idx_arr = self.image_id(latents_bases, posX_id_arr, posY_id_arr, orientation_arr, scale_id_arr)
        treatment = imgs[image_idx_arr].reshape((n_sample, 64 * 64)).astype(np.float64)
        treatment += rng.normal(0.0, 0.1, treatment.shape)
        latent_feature = latents_values[image_idx_arr]  # (color, shape, scale, orientation, posX, posY)
        treatment_proxy = latent_feature[:, 2:5]  # (scale, orientation, posX)

        posX_id_proxy = np.array([16] * n_sample)
        scale_id_proxy = np.array([3] * n_sample)
        orientation_proxy = np.array([0] * n_sample)
        proxy_image_idx_arr = self.image_id(latents_bases, posX_id_proxy, posY_id_arr, orientation_proxy, scale_id_proxy)
        outcome_proxy = imgs[proxy_image_idx_arr].reshape((n_sample, 64 * 64)).astype(np.float64)
        outcome_proxy += rng.normal(0.0, 0.1, outcome_proxy.shape)

        structural = self.structural_func(treatment, weights)
        outcome = structural * (posY_id_arr - 15.5) ** 2 / 85.25 + rng.normal(0.0, 0.5, n_sample)
        outcome = outcome[:, np.newaxis]
        
        do_A, EY_do_A = self.generate_test_dsprite(data_path)
        return treatment, outcome, treatment_proxy, outcome_proxy, do_A, EY_do_A

















