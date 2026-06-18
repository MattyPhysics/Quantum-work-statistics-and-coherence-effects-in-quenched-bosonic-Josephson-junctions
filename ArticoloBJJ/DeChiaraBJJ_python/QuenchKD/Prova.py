import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lpmv, gammaln


def ClassicalBound(alpha, deltaphi, U_i, U_f, N, r, n_vec, index_sign=1):
    beta_mod = np.sqrt(1 - alpha**2)
    c = np.array([alpha, beta_mod * np.exp(1j * deltaphi)], dtype=complex)
    
    omega_i = np.sqrt(N * U_i + 1)
    omega_f = np.sqrt(N * U_f + 1)
    
    m_max = 50 # Limita il range se necessario per evitare overflow
    m_vec = np.arange(0, m_max + 1, 2)
    
    term_contribution = 0
    
    for m in m_vec:
        E_f_val = (m + 0.5) * omega_f
        
        # Calcolo ampiezze con gestione errori
        S_ik = np.array([Amplitudes(m, n, r, index_sign) for n in n_vec])
        
        # Se le ampiezze sono nan, saltiamo questa iterazione
        if np.any(np.isnan(S_ik)):
            continue
            
        p_j = (S_ik**2) * (np.abs(c)**2)
        
        # Calcolo interferenza
        p1_val = np.sum(p_j) + 2 * S_ik[0] * S_ik[1] * np.real(c[1] * np.conj(c[0]))
        
        # Calcolo contributi
        for i_n in range(len(n_vec)):
            E_i_val = (n_vec[i_n] + 0.5) * omega_i
            if E_i_val > E_f_val:
                # np.sqrt(p1 * p_j) può dare NaN se il prodotto è negativo 
                # (es. a causa di errori di precisione)
                arg = np.abs(p1_val * p_j[i_n])
                term_contribution += (E_i_val - E_f_val) * np.sqrt(arg)
                
    return term_contribution


def Amplitudes(m, n, r, index_sign):
    if (m + n) % 2 != 0: 
        return 0
    l = (m + n) // 2
    k = (m - n) // 2
    x=1.0/np.cosh(np.abs(r))
    legendrevalue = lpmv(abs(k), l, x)
    
    log_pref = 0.5 * (gammaln(min(n, m) + 1) - gammaln(max(n, m) + 1))
    
    if m >= n:
        phase = (index_sign)**abs(k)
    else:
        phase = (-index_sign)**abs(k)
        
    return np.exp(log_pref) * np.sqrt(x) * legendrevalue * phase

# 1. Parametri fissi come da tua richiesta
U_f = 0.0
U_i = 0.1
alpha = 0.3
N = 100
n_vec = np.array([0, 2])

# Calcolo costanti per questi parametri
omega_i = np.sqrt(N * U_i + 1)
omega_f = np.sqrt(N * U_f + 1)
r = 0.5 * np.log(omega_f / omega_i)
index_sign = np.sign(omega_i - omega_f)

# 2. Vettore di variazione per delta_phi
delta_phi_vec = np.linspace(0, 2 * np.pi, 200)
bound_values = []

# 3. Calcolo del bound per ogni delta_phi
for phi in delta_phi_vec:
    val = ClassicalBound(alpha, phi, U_i, U_f, N, r, n_vec, index_sign)
    bound_values.append(val)
import numpy as np
import matplotlib.pyplot as plt

# 1. Definizione dei parametri fissi e vettori di variazione
U_f = 0.0
U_i = 0.1
N = 100
n_vec = np.array([0, 2])

# Calcolo costanti globali
omega_i = np.sqrt(N * U_i + 1)
omega_f = np.sqrt(N * U_f + 1)
r = 0.5 * np.log(omega_f / omega_i)
index_sign = np.sign(omega_i - omega_f)

# Vettori per gli assi
alpha_vec = np.linspace(0.01, 0.99, 50)  # Evitiamo 0 o 1 per evitare instabilità ai bordi
phi_vec = np.linspace(0, 2 * np.pi, 50)

# 2. Creazione della griglia e calcolo
Alpha_grid, Phi_grid = np.meshgrid(alpha_vec, phi_vec)
bound_matrix = np.zeros_like(Alpha_grid)

for i in range(len(phi_vec)):
    for j in range(len(alpha_vec)):
        bound_matrix[i, j] = ClassicalBound(alpha_vec[j], phi_vec[i], 
                                            U_i, U_f, N, r, n_vec, index_sign)

# 3. Plot della Heatmap
plt.figure(figsize=(9, 6))
# Usiamo origin='lower' per avere (0,0) in basso a sinistra
plt.imshow(bound_matrix, extent=[0, 1, 0, 2 * np.pi], origin='lower', 
           aspect='auto', cmap='inferno', interpolation='bilinear')

plt.colorbar(label='Classical Bound')
plt.xlabel('Alpha')
plt.ylabel('Delta Phi')
plt.title(f'Heatmap Classical Bound (Ui={U_i}, Uf={U_f})')
plt.show()
# 4. Plot della sezione
plt.figure(figsize=(8, 5))
plt.plot(delta_phi_vec, bound_values, label=f'Alpha={alpha}, Ui={U_i}', color='blue', lw=2)
plt.axhline(0, color='black', linestyle='--', alpha=0.3)
plt.xlabel(r'$\Delta\phi$')
plt.ylabel('Classical Bound')
plt.title(f'Sezione del Classical Bound (U_i={U_i}, U_f={U_f}, Alpha={alpha})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()