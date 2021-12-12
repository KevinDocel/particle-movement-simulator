import os
import pickle
import platform

import numpy as np
from tqdm import tqdm

import constant as C
import function as F


def merge(x, n_merged, min_dist, min_pos):
    
    def _can_merge(x, n_merged, min_dist):
        valid_idx = np.argwhere(n_merged > 0)

        for i in range(len(valid_idx) - 1):
            idx_i = valid_idx[i]
            idx_i_plus = valid_idx[i+1]
            
            if x[idx_i] + min_dist >= x[idx_i_plus]:
                return True
        
        return False

    # merge once
    def _merge(x, n_merged, min_dist):
        valid_idx = np.argwhere(n_merged > 0)

        for i in range(len(valid_idx) - 1):
            idx_i = valid_idx[i]
            idx_i_plus = valid_idx[i+1]

            if x[idx_i] + min_dist >= x[idx_i_plus]:
                x_new_i = np.average([x[idx_i], x[idx_i_plus]], 
                                        weights=[n_merged[idx_i], n_merged[idx_i_plus]])
                
                x[idx_i] = x_new_i
                x[idx_i_plus] = x_new_i
                n_merged[idx_i] = 0
                n_merged[idx_i_plus] += n_merged[idx_i]

        return x, n_merged

    # 0. merge <=0 into 1 particle
    lt_zero_idx = np.argwhere(x <= min_pos)
    x[lt_zero_idx] = min_pos
    num_lt_zero = np.sum(n_merged[lt_zero_idx])
    n_merged[lt_zero_idx] = 0
    n_merged[lt_zero_idx[0]] = num_lt_zero

    # 1. merge others
    while _can_merge(x, n_merged, min_dist):
        x, n_merged = _merge(x, n_merged, min_dist)
    
    return x, n_merged



def main():
    n_total = 4307
    delta_t = 0.1 # 0.001
    max_t = 120
    x_0_start = 0.0005
    x_0_end = 0.008

    min_pos = x_0_start
    n_t = int(max_t // delta_t)
    # x = np.linspace(x_0_start, x_0_end, n_total, endpoint=True)
    # merged_into_idxs = np.arange(n_total)
    # n_merged = np.ones(n_total, dtype=np.int64)
    # valid_idx = np.arange(n_total)

    x_mat = np.zeros((n_t + 1, n_total), dtype=np.float32)
    x_mat[0] = np.linspace(x_0_start, x_0_end, n_total, endpoint=True)
    # pre_mat = np.ones((n_t + 1, n_total), dtype=np.int64)
    # pre_mat[0] = merged_into_idxs
    n_merged_mat = np.zeros((n_t + 1, n_total), dtype=np.int32)
    n_merged_mat[0] = np.ones(n_total, dtype=np.int32)
    
    if platform.system() == "Windows":
        cmd_clear = "cls"
    else:
        cmd_clear = "clear"
    
    for t in range(n_t):
        os.system(cmd_clear)
        print(f"********** Iteration {t+1}/{n_t} **********")
        
        x = x_mat[t].copy()
        n_merged = n_merged_mat[t].copy()
        # 0. remove invalid particles, including merged and position <= 0
        # valid_idx = np.argwhere((n_merged > 0) & (x > 0))
        valid_idx = np.argwhere(n_merged > 0)
        
        if len(valid_idx) == 1:
            assert x[valid_idx] == min_pos
            break

        x = x[valid_idx].flatten()
        n_merged = n_merged[valid_idx].flatten()

        n = len(x)

        # 1. new position at next time t
        print("Computing new positions ...")
        for i in tqdm(range(n)):
            idx = np.asarray([j for j in range(n) if j != i], dtype=np.int64)

            f_dd = np.zeros(n)
            f_dd[idx] = F.F_DD_vec(x[idx], n_merged[idx], x[i], n_merged[i])

            d = np.ones(n) * C.KB * C.T / (6 * C.PI * C.A * C.ETA)
            d[idx] = F.D_vec(x[idx], x[i])
            f_m = F.F_M_vec(x, n_merged)
            d_f_m = d * f_m
            
            # x[i] = x[i] + 1/C.GAMMA * np.sum(f_dd) * delta_t + np.sum(d_f_m) * delta_t / (C.KB * C.T)
            x[i] = x[i] + 1/F.GAMMA(n_merged[i]) * np.sum(f_dd) * delta_t + np.sum(d_f_m) * delta_t / (C.KB * C.T)


        # 2. add random noise from Gaussian
        print("Add random gaussian noise ...")
        # delta_x_g = np.random.normal(0, np.sqrt(2 * C.D * delta_t), n)
        # x += delta_x_g
        gt_zero_idx = np.argwhere(x > min_pos).flatten()
        delta_x_g = np.random.normal(0, np.sqrt(2 * C.D * delta_t), len(gt_zero_idx))
        x[gt_zero_idx] += delta_x_g

        # 3. merge
        print("Merge particles ...")
        x, n_merged = merge(x, n_merged, 2 * C.A, min_pos)
        
        # 4. prepare for plot
        x_mat[t+1][valid_idx] = x.copy().reshape((-1, 1))
        n_merged_mat[t+1][valid_idx] = n_merged.copy().reshape((-1, 1))


    print("Saving ...")
    with open(f"dxw_delta_t_{delta_t}.pkl", "wb") as f:
        pickle.dump({
            "X": x_mat,
            "N": n_merged_mat,
            "delta_t": delta_t,
            "max_t": max_t,
            "x_0_start": x_0_start,
            "x_0_end": x_0_end,
        }, f)


if __name__ == "__main__":
    main()
