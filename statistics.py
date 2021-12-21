import pickle

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def load(filepath):
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    
    return data

def main():
    filename = "dxw_delta_t_0.001_1220"
    n_t = 10
    n_total = []
    n_total_merged = []

    data = load(f"{filename}.pkl")
    delta_t = data["delta_t"]
    N_mat = data["N"]
    
    for i in tqdm(range(N_mat.shape[0])):
        n = N_mat[i]
        n_total.append(np.sum(n)- n[0])
        n_total_merged.append(np.sum(n > 0) - 1)
    
    for i in range(0, N_mat.shape[0], int(n_t // delta_t)):
        print(f"{i*delta_t:3.0f} s, n_total {n_total[i]:4.0f}, n_total_merged {n_total_merged[i]:4.0f}")


if __name__  == "__main__":
    main()