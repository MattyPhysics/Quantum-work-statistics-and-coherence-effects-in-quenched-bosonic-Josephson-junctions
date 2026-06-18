import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

def twomode_bose_hubbard_hamiltonian(N, K, U):
    """
    Constructs the two-mode Bose-Hubbard Hamiltonian matrix in the Fock basis (left-right basis)
    
    Parameters:
    - N: Total number of bosons
    - K: Tunneling coefficient
    - U: Self-interaction coefficient
    
    Returns:
    - H: The Hamiltonian matrix 
    """
    # Label of basis states: 0 to N bosons in mode 1 (left well) , rest in mode 2 (right well)
    dim = N + 1
    
    # Initialize the Hamiltonian matrix
    H = np.zeros((dim, dim))
    
    # Interaction term (dagonal): U/4 *(n1-n2)^2 
    for n1 in range(N + 1):
        n2 = N - n1
        interaction_energy = (U / 4) * (n1 - n2)**2
        H[n1, n1] = interaction_energy
    
    # Tunneling term: -k/2 * (a1^dagger a2 + a2^dagger a1)
    for n1 in range(1, N + 1):
        # Off-diagonal tunneling elements
        H[n1, n1 - 1] = H[n1 - 1, n1] = -K/2 * np.sqrt(n1 * (N - n1 + 1))
    
    return H

##############################################################################################################################################

# Parameters initialization
N = 100 # Number of particles
K = 10   # Tunneling strength 
U_f=0*K
# Construct the final Hamiltonian
H_f = twomode_bose_hubbard_hamiltonian(N, K, U_f)
# Final plasma frequency
omega_f=K*(U_f*N/K+1)**0.5

# In the following we study the energy spectrum of H_f
final_eigenvalues, final_eigenvectors = eigh(H_f)
spectrum_label=np.arange(0,final_eigenvalues.shape[0],1)
plt.plot(spectrum_label,(final_eigenvalues-final_eigenvalues[0])/omega_f/N,'o',markersize=1,color='blue')
plt.xlim(0,100)
plt.ylim(0)
plt.show()
print(final_eigenvalues[2],final_eigenvalues[0],final_eigenvalues[2]-final_eigenvalues[1],omega_f)

##############################################################################################################################################
# Possible initial values of the self interaction term
U_range=np.geomspace(pow(10,-3)*K,pow(10,1)*K,100)

# Evaluation of the average work and the real part of the variance for a sudden quench
W=[]
W_analytic=[]
W_squared_analytic=[] # second moment of work
W_var=[]
W_var_analytic=[]

for U_i in U_range:
    H_i = twomode_bose_hubbard_hamiltonian(N, K, U_i)
    omega_i = K * (U_i * N / K + 1)**0.5
    
    eigenvalues, eigenvectors = eigh(H_i)
    ground_state = eigenvectors[:, 0]
    second_exited = eigenvectors[:, 2] 
    

    # We need to make sure that the value of the eigenstate of the ground state of H_i is positive in the origine of the 
    # x=(n1-n2)/2 axis
    origine = N// 2
    
    if ground_state[origine] < 0:
        ground_state = -ground_state
        
    # The second exited state must be negative in yhe origine
    if second_exited[origine] >0:
        second_exited = -second_exited
        
    # Initial superposition
    initial_state = 1/np.sqrt(2) * (ground_state -second_exited)

    # Evaluation of the average work for every U_i
    W.append(np.dot(initial_state,H_f.dot(initial_state))-np.dot(initial_state,H_i.dot(initial_state)))
    W_analytic.append(N/4*(U_f-U_i)*K/omega_i*(3-np.sqrt(2)))
    
    # Evaluation of the variance
    work_op=H_f-H_i
    work_op_squared=work_op@work_op
    W_var.append(np.dot(initial_state,work_op_squared.dot(initial_state))-np.dot(initial_state,work_op.dot(initial_state))**2)
    W_var_analytic.append((N*(U_f-U_i)*K/omega_i)**2/16*(21-6*np.sqrt(2))-(N/4*(U_f-U_i)*K/omega_i*(3-np.sqrt(2)))**2)


# Array for the plots
Average_quantum_work=np.array(W)
Average_quantum_work_analytic=np.array(W_analytic)
Work_variance=np.array(W_var)
Work_variance_analytic=np.array(W_var_analytic)

# Plot of the average quantum work
plt.plot(U_range/K, np.abs(Average_quantum_work/N/K),'o',markersize=2,label='Numerical',color='blue')
plt.plot(U_range/K, np.abs(Average_quantum_work_analytic/K/N),label='Analytical',color='red')
plt.xlabel('Ui/K')
plt.ylabel('<W>quantum/K ')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()

#Plot of the work variance
plt.plot(U_range/K, Work_variance/(K**2),'o',markersize=2,label='Numerical',color='blue')
plt.plot(U_range/K, Work_variance_analytic/(K**2),label='Analytical',color='red')
plt.xlabel('U_f-Ui/J')
plt.ylabel('var(W)/J^2 ')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()
