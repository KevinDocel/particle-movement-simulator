import pickle

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def load(filepath):
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    
    return data


def plot(X, N, delta_t, x_max, y_max, filepath, dpi):
    plt.figure()
    plt.xlim([0, x_max])
    plt.ylim([0, y_max])

    for i in tqdm(range(X.shape[1])):
        x_i = []
        t_i = []
        for t in range(X.shape[0]):
            if N[t, i] >= 0:
                x_i.append(X[t, i])
                t_i.append(t)

            if N[t, i] == 0:
                break
        
        t_i = np.asarray(t_i) * delta_t
        plt.plot(t_i, x_i, 'k-')
    
    plt.savefig(filepath, dpi=dpi)

def main():
    filename = "dxw_delta_t_0.001_1220"
    save_format = "svg"
    dpi = 600

    print("Reading data ...")
    data = load(f"{filename}.pkl")

    print("Plotting ...")
    plot(data["X"], data["N"], data["delta_t"], data["max_t"], data["x_0_end"], f"{filename}.{save_format}", dpi)

if __name__ == "__main__":
    main()