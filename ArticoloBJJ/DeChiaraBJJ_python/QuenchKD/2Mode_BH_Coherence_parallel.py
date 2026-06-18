import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import itertools
import time

def twomode_bose_hubbard_hamiltonian_vectorized(N, K, U):
    dim = N + 1
    n1 = np.arange(dim)
    n2 = N - n1
    diag = (U / 4.0) * (n1 - n2) ** 2
    H = np.diag(diag)
    k_idx = np.arange(1, dim)
    off_vals = -0.5 * K * np.sqrt(k_idx * (N - k_idx + 1))
    H += np.diag(off_vals, k=1) + np.diag(off_vals, k=-1)
    return H

def worker(args):
    sigma, U_i, N, K, U_f, H_f = args
    # Build initial Hamiltonian and diagonalize
    H_i = twomode_bose_hubbard_hamiltonian_vectorized(N, K, U_i)
    eigenvalues, eigenvectors = eigh(H_i)

    # ground state and second excited state
    ground_state = eigenvectors[:, 0].copy()
    second_exited = eigenvectors[:, 2].copy()

    origine = N // 2
    # Fix phases
    if ground_state[origine] < 0:
        ground_state = -ground_state
    if second_exited[origine] > 0:
        second_exited = -second_exited

    # prepare initial superposition
    initial_state = (ground_state + sigma * second_exited) / np.sqrt(2.0)

    # compute works and variances
    omega_i = K * np.sqrt(U_i * N / K + 1.0)
    work_op = H_f - H_i
    exp_Hf = initial_state @ (H_f @ initial_state)
    exp_Hi = initial_state @ (H_i @ initial_state)
    w = exp_Hf - exp_Hi
    w_analytic = (N / 4.0) * (U_f - U_i) * K / omega_i * (3.0 + sigma * np.sqrt(2.0))

    work_op_squared = work_op @ work_op
    var_w = initial_state @ (work_op_squared @ initial_state) - (initial_state @ (work_op @ initial_state)) ** 2

    term_x4 = 21.0 + sigma * 6.0 * np.sqrt(2.0)
    term_x2_squared = (3.0 + sigma * np.sqrt(2.0)) ** 2
    var_w_analytic = (N * (U_f - U_i) * K / omega_i) ** 2 / 16.0 * (term_x4 - term_x2_squared)

    return (w, w_analytic, var_w, var_w_analytic)

if __name__ == "__main__":
    # Parameters
    N = 100
    K = 10.0
    U_f =  0*K

    # Precompute final Hamiltonian and omega_f (H_f reused by workers)
    H_f = twomode_bose_hubbard_hamiltonian_vectorized(N, K, U_f)
    omega_f = K * np.sqrt(U_f * N / K + 1.0)

    # U range and signs
    U_range = np.geomspace(1e-3* K, 1e3* K, 100)
    signs_list = [1, -1]
    labels = ['+', '-']
    colors_num = ['blue', 'green']
    colors_ana = ['red', 'orange']

    # Build tasks: product(sign, U_i)
    tasks = [(sigma, U_i, N, K, U_f, H_f) for sigma in signs_list for U_i in U_range]

    start = time.time()
    # Parallel execution
    with ProcessPoolExecutor() as exe:
        results = list(exe.map(worker, tasks))
    elapsed = time.time() - start
    print(f"Parallel computation for a system of {N} atoms finished in {elapsed:.2f} s for {len(U_range)} points and {len(signs_list)} signs.")

    # Reshape results to (n_signs, n_U, 4)
    results = np.array(results, dtype=float).reshape(len(signs_list), len(U_range), 4)

    # Prepare plots
    fig1, ax1 = plt.subplots(figsize=(9, 7))
    fig2, ax2 = plt.subplots(figsize=(9, 7))

    # --- PLOT CURVES ---
    for idx, sigma in enumerate(signs_list):
        W = results[idx, :, 0]
        W_analytic = results[idx, :, 1]
        Var = results[idx, :, 2]
        Var_analytic = results[idx, :, 3]

        ax1.plot(U_range / K, np.abs(W / N / K), 'o', markersize=2, label=f'Bose-Hubbard ({labels[idx]})', color=colors_num[idx])
        ax1.plot(U_range / K, np.abs(W_analytic / K / N), label=f'Analytical ({labels[idx]})', color=colors_ana[idx], linestyle='--')

        ax2.plot(U_range / K, Var / (K ** 2), 'o', markersize=2, label=f'Bose-Hubbard ({labels[idx]})', color=colors_num[idx])
        ax2.plot(U_range / K, Var_analytic / (K ** 2), label=f'Analytical ({labels[idx]})', color=colors_ana[idx], linestyle='--')

    # --- COMMON STYLING (Backgrounds & Lines) ---
    x_sep = U_f / K
    x_min_data = np.min(U_range) / K
    x_max_data = np.max(U_range) / K

    # Applico le decorazioni a entrambi gli assi
    for ax in [ax1, ax2]:
        # 1. Linea verticale tratteggiata
        ax.axvline(x=x_sep, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
        
        # 2. Sfondo colorato esteso (per coprire il padding)
        ax.axvspan(1e-9, x_sep, color='azure', alpha=0.6)       # Sinistra
        ax.axvspan(x_sep, 1e9, color='mistyrose', alpha=0.4)    # Destra

        # 3. Impostazioni scale e tick
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(axis='both', labelsize=20)
        
        # 4. Elimino le strisce bianche forzando i limiti sui dati
        ax.set_xlim(x_min_data, x_max_data)

    # --- SPECIFIC LABELS ---
    # Work Plot
    ax1.set_xlabel(r'$U_i/K$', fontsize=20)
    ax1.set_ylabel(r'$|\langle w\rangle|/K$', fontsize=20)
    ax1.legend(loc='lower left', fontsize=15)

    # Variance Plot
    ax2.set_xlabel(r'$U_i/K$', fontsize=20)
    ax2.set_ylabel(r'$\mathrm{Re}\left[(\Delta w)^2\right]/K^2$', fontsize=20)
    ax2.legend(loc='upper left', fontsize=15)

    plt.show()