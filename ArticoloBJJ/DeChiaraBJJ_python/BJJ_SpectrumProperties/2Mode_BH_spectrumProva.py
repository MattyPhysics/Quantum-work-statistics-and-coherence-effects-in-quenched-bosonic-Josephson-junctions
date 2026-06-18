import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

def twomode_bose_hubbard_hamiltonian(N, K, U):
    """
    Constructs the two-mode Bose-Hubbard Hamiltonian for a system of N bosons.
    
    Parameters:
    - N: Total number of bosons
    - K: Tunneling strength
    - U: Self-interaction strength
    
    Returns:
    - H: The Hamiltonian matrix
    """
    # Number of basis states (0 to N bosons in mode 1, rest in mode 2)
    dim = N + 1
    
    # Initialize the Hamiltonian matrix
    H = np.zeros((dim, dim))
    
    # Interaction term: U/2 * n(n-1) for both modes
    for n1 in range(N + 1):
        n2 = N - n1
        interaction_energy = (U / 4) * (n1 - n2)**2
        H[n1, n1] = interaction_energy
    
    # Tunneling term: -k/2 * (a1^dagger a2 + a2^dagger a1)
    for n1 in range(1, N + 1):
        # Off-diagonal tunneling elements
        H[n1, n1 - 1] = H[n1 - 1, n1] = -K/2 * np.sqrt(n1 * (N - n1 + 1))
    
    return H


def atomnumberdifference(N):
    dim = N + 1
    n = np.zeros((dim, dim))
    for i in range(dim):
        n[i, i] = i - N/2
    return n

def coherence(N):
    dim = N + 1
    alpha = np.zeros((dim, dim))
    for n1 in range(1, N + 1):
        alpha[n1, n1 - 1] = alpha[n1 - 1, n1] = 1/2 * np.sqrt(n1 * (N - n1 + 1))
    return alpha

# Parametri
N = 100
N_grid=1000
K = 1
# Generiamo U/K da 10^-5 a 10^5 per centrare i confini riscalati
U_range = np.geomspace(1e-5, 1e5, 100)
variance_n = []
average_coherence = []
U_f=0.01*K
# Construct the final Hamiltonian
H_f = twomode_bose_hubbard_hamiltonian(N, K, U_f)
# Final plasma frequency
omega_f=K*(U_f*N/K+1)**0.5


# In the following we study the energy spectrum of H_f
final_eigenvalues, final_eigenvectors = eigh(H_f)
spectrum_label=np.arange(0,final_eigenvalues.shape[0],1)
plt.figure(figsize=(10,8))
plt.plot(spectrum_label,(final_eigenvalues+K/2*(N+1))/omega_f,'o',markersize=2,color='blue')
plt.xlim(0,101)
plt.ylim(0)
plt.xlabel(r'Eigenvalue index $[i]$', fontsize=20)
plt.ylabel(r'$E\left[i\right]/\omega_f$', fontsize=20)
plt.tick_params(axis='both', labelsize=20)
plt.show()