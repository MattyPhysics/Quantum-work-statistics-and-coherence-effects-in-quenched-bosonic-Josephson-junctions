import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
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
    U_i, N, K, U_f, H_f = args
    sigma = 1 
    H_i = twomode_bose_hubbard_hamiltonian_vectorized(N, K, U_i)
    eigenvalues, eigenvectors = eigh(H_i)

    ground_state = eigenvectors[:, 0].copy()
    second_exited = eigenvectors[:, 2].copy()
    origine = N // 2
    if ground_state[origine] < 0: ground_state = -ground_state
    if second_exited[origine] > 0: second_exited = -second_exited

    alpha = 1.0
    beta = np.sqrt(1 - alpha**2)
    initial_state = alpha * ground_state + sigma * beta * second_exited
    initial_state /= np.linalg.norm(initial_state)

    omega_i = K * np.sqrt(U_i * N / K + 1.0)
    work_op = H_f - H_i
    exp_Hf = initial_state @ (H_f @ initial_state)
    exp_Hi = initial_state @ (H_i @ initial_state)
    w = exp_Hf - exp_Hi
    
    w_D = (N / 2.0) * ((U_f - U_i) / omega_i) * K * (2 * beta**2 + 0.5)
    w_C = (N * np.sqrt(2) / 2.0) * ((U_f - U_i) / omega_i) * K * alpha * beta * sigma
    w_analytic = w_C + w_D
    
    work_op_squared = work_op @ work_op
    var_w = initial_state @ (work_op_squared @ initial_state) - (initial_state @ (work_op @ initial_state)) ** 2
    var_w_analytic = 3/16*N**2*(U_f-U_i)**2/(omega_i)**2*K**2*(1+12*beta**2)+3*np.sqrt(2)/4*N**2*(U_f-U_i)**2/(omega_i)**2*K**2*alpha*beta*sigma - w_analytic**2

    return (w, w_analytic, var_w, var_w_analytic)

if __name__ == "__main__":
    N, K = 100, 1.0
    U_f = 0.0 * K
    H_f = twomode_bose_hubbard_hamiltonian_vectorized(N, K, U_f)
    U_range = np.geomspace(1e-3 * K, 1e3 * K, 100)
    tasks = [(U_i, N, K, U_f, H_f) for U_i in U_range]

    with ProcessPoolExecutor() as exe:
        results = np.array(list(exe.map(worker, tasks)))

    
    plt.figure(figsize=(9, 6))
    plt.plot(U_range[::2] / K, np.abs(results[::2, 1] / K), label='Analytical', color='red', linestyle='-', linewidth=3.5)
    plt.plot(U_range[::2] / K, np.abs(results[::2, 0] / K), 'o', markersize=3.5, label='Bose-Hubbard', color='blue')
    plt.axvline(x=1e2, color='black', linestyle='--', linewidth=2.5, alpha=0.7)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\frac{U_i}{K}$', fontsize=30)
    plt.ylabel(r'$\frac{\mathcal{W}}{K}$', fontsize=30)
    plt.tick_params(axis='both', labelsize=20)
    plt.legend(loc='upper left', fontsize=25)  
    plt.tight_layout()
    plt.savefig('AverageWork_VaryingUinitial_GS.pdf', format='pdf', bbox_inches='tight')

    plt.figure(figsize=(9, 6))
    plt.plot(U_range[::2] / K, results[::2, 3] / K**2, label='Analytical', color='red', linestyle='-', linewidth=3.5)
    plt.plot(U_range[::2] / K, results[::2, 2] / K**2, 'o', markersize=3.5, label='Bose-Hubbard', color='blue')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\frac{U_i}{K}$', fontsize=30)
    plt.ylabel(r'$Re[(\Delta \mathcal{W})^2]/{K^2}$', fontsize=30)
    plt.tick_params(axis='both', labelsize=22)
    plt.legend(loc='upper left', fontsize=25) 
    plt.savefig('WorkVariance_VaryingUinitial_GS.pdf', format='pdf', bbox_inches='tight')

    plt.show()