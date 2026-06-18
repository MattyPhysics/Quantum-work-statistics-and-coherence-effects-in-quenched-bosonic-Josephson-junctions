import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from scipy.special import lpmv, gammaln
import matplotlib.ticker as ticker

# =============================================================================
#  FUNZIONI FISICHE E MATEMATICHE
# =============================================================================

def twomode_bose_hubbard_hamiltonian(N, K, U):
    """Costruisce l'Hamiltoniana di Bose-Hubbard a due modi."""
    dim = N + 1
    H = np.zeros((dim, dim))
    # Termine di interazione
    for n1 in range(N + 1):
        n2 = N - n1
        H[n1, n1] = (U / 4) * (n1 - n2)**2
    # Termine di tunneling
    for n1 in range(1, N + 1):
        H[n1, n1 - 1] = H[n1 - 1, n1] = -K/2 * np.sqrt(n1 * (N - n1 + 1))
    return H

def SqueezingMatrixTheory(m, n, x, index_sign):
    """Calcola l'elemento di matrice teorico S_mn usando i polinomi di Legendre."""
    if (m + n) % 2 != 0: return 0
    l = (m + n) // 2
    k = (m - n) // 2
    
    # lpmv(m, v, x) corrisponde a P_v^m(x)
    legendrevalue = lpmv(abs(k), l, x)
    
    # Calcolo prefattore logaritmico per stabilità numerica
    log_pref = 0.5 * (gammaln(min(n, m) + 1) - gammaln(max(n, m) + 1))
    
    # Fase dipendente dalla direzione del quench
    if m >= n:
        phase = (-index_sign)**abs(k)
    else:
        phase = (index_sign)**abs(k)
        
    return np.exp(log_pref) * np.sqrt(x) * legendrevalue * phase

# =============================================================================
#  PARAMETRI E SIMULAZIONE
# =============================================================================

# Parametri modello
N = 100
K = 1.0
U_i = 0.0
U_f = 0.1*K

# Diagonalizzazione
H_i = twomode_bose_hubbard_hamiltonian(N, K, U_i)
H_f = twomode_bose_hubbard_hamiltonian(N, K, U_f)
E_i, V_i = eigh(H_i)
E_f, V_f = eigh(H_f)

# Stato iniziale |Psi> = (|0> + |2>) / sqrt(2)
c_coeffs = np.zeros(N + 1, dtype=complex)
c_coeffs[0] = 1.0 / np.sqrt(2)
c_coeffs[2] = -1.0 / np.sqrt(2)
rho_0 = np.outer(c_coeffs, np.conj(c_coeffs))

# Parametri fisici (frequenze e Bogoliubov)
omega_i = K * np.sqrt(U_i * N / K + 1) 
omega_f = K * np.sqrt(U_f * N / K + 1) 
index_sign = np.sign(omega_f - omega_i)

Q_param = (omega_i**2 + omega_f**2) / (2 * omega_i * omega_f)
u_bog = np.sqrt((Q_param + 1) / 2.0)
x_leg = 1.0 / u_bog

# Matrice di Squeezing Numerica
SqueezingMatrixNum = V_f.T @ V_i 

# Matrici per salvare i risultati (m x n)
q_num_matrix = np.zeros((N + 1, N + 1), dtype=complex)
q_theory_matrix = np.zeros((N + 1, N + 1), dtype=complex)

# Calcolo KDQ per n=0 e n=2
for n in [0, 2]:
    for m in range(0, N+ 1, 2):
        # --- Numerica ---
        interf_num = sum(rho_0[n, s] * np.conj(SqueezingMatrixNum[m, s]) for s in [0, 2])
        q_num_matrix[m, n] = SqueezingMatrixNum[m, n] * interf_num
        
        # --- Teorica ---
        interf_theory = sum(rho_0[n, s] * SqueezingMatrixTheory(m, s, x_leg, index_sign) for s in [0, 2])
        q_theory_matrix[m, n] = SqueezingMatrixTheory(m, n, x_leg, index_sign) * interf_theory

# =============================================================================
#  CALCOLO DEL NONPOSITIVITY FUNCTIONAL (N)
# =============================================================================
# Definizione Cap. 3: N = -1 + Sum(|q_mn|)
# Sommiamo i valori assoluti di tutta la matrice teorica (gli elementi non calcolati sono 0)
sum_abs_q = np.sum(np.abs(q_theory_matrix))
nonpositivity_val = sum_abs_q - 1.0

# =============================================================================
#  CALCOLO DEL LAVORO MEDIO <W>
# =============================================================================

# 1. Numerico
W_avg_num = 0
for n in [0, 2]:
    for m in range(N + 1):
        work_val = E_f[m] - E_i[n]
        W_avg_num += q_num_matrix[m, n].real * work_val

# 2. Teorico (Formula analitica)
alpha_sq = c_coeffs[0]**2; beta_sq = c_coeffs[2]**2; 
j_idx = 0; k_idx = 2
prefactor = (omega_f**2 - omega_i**2) / (4 * omega_i)
term_tpm = alpha_sq * (2 * j_idx + 1) + beta_sq * (2 * k_idx + 1)
term_coh = 2 * c_coeffs[0]*c_coeffs[2] * np.sqrt((j_idx + 1) * (j_idx + 2))
W_avg_theory = np.real(prefactor * (term_tpm + term_coh))

# =============================================================================
#  PREPARAZIONE DATI PER IL PLOT
# =============================================================================

def E_th_func(k, omega):
    return omega * (k + 0.5)

m_max_plot = 40 

data_0 = {'W_num': [], 'q_num': [], 'W_th': [], 'q_th': []}
data_2 = {'W_num': [], 'q_num': [], 'W_th': [], 'q_th': []}

for n, dataset in zip([0, 2], [data_0, data_2]):
    for m in range(0, m_max_plot, 2):
        w_n = E_f[m] - E_i[n]
        val_q = np.real(q_num_matrix[m, n])
        if abs(val_q) > 1e-10:
            dataset['W_num'].append(w_n)
            dataset['q_num'].append(val_q)
            
        w_t = E_th_func(m, omega_f) - E_th_func(n, omega_i)
        val_qt = np.real(q_theory_matrix[m, n])
        if abs(val_qt) > 1e-10:
            dataset['W_th'].append(w_t)
            dataset['q_th'].append(val_qt)

for d in [data_0, data_2]:
    for key in d:
        d[key] = np.array(d[key])

# =============================================================================
#  PLOT PRINCIPALE
# =============================================================================

fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')

# Barre Numeriche
ax.bar(data_0['W_num'], data_0['q_num'], width=0.3, color='blue', alpha=0.5, 
       edgecolor='blue', linewidth=0.5, label=r'Numerical $q_{m0}$')
ax.bar(data_2['W_num'], data_2['q_num'], width=0.3, color='green', alpha=0.5, 
       edgecolor='green', linewidth=0.5, label=r'Numerical $q_{m2}$')

# Stem Teorici
markerline0, stemlines0, baseline0 = ax.stem(
    data_0['W_th'], data_0['q_th'], 
    linefmt='r-', markerfmt='ro', basefmt=" ", 
    label='Theory'
)
markerline2, stemlines2, baseline2 = ax.stem(
    data_2['W_th'], data_2['q_th'], 
    linefmt='r-', markerfmt='ro', basefmt=" "
)
plt.setp([markerline0, markerline2], markersize=4)
plt.setp([stemlines0, stemlines2], linewidth=1.2, alpha=0.8)
plt.setp([baseline0, baseline2], visible=False)

ax.axhline(0, color='black', linewidth=1.2)
ax.set_xlabel(r'$W$', fontsize=20)
ax.set_ylabel(r'$P(W)$', fontsize=20)
ax.set_xlim(-5,35)
ax.set_ylim(-0.15,0.8)
ax.grid(True, alpha=0.15, linestyle='--')
ax.legend(loc='upper left', fontsize=13)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.tick_params(axis='both', labelsize=20)

# --- INSETS ---
inset_w = 0.22  
inset_h = 0.28  
pos_x1 = 0.44   
pos_x2 = 0.76   
pos_y = 0.70    

ax_ins1 = ax.inset_axes([pos_x1, pos_y, inset_w, inset_h])
ax_ins2 = ax.inset_axes([pos_x2, pos_y, inset_w, inset_h])
n_points = 10 

# Inset 1
ax_ins1.bar(data_0['W_num'][:n_points], data_0['q_num'][:n_points], width=0.4, 
            color='blue', alpha=0.6, edgecolor='black', linewidth=0.3)
markerline, stemlines, baseline = ax_ins1.stem(
    data_0['W_th'][:n_points], data_0['q_th'][:n_points], 
    linefmt='r-', markerfmt='ro', basefmt=" "
)
plt.setp(stemlines, linewidth=0.8); plt.setp(markerline, markersize=2); plt.setp(baseline, visible=False)
ax_ins1.axhline(0, color='black', linewidth=0.5)
ax_ins1.set_xlabel(r'$W$', fontsize=20, labelpad=6) 
ax_ins1.tick_params(axis='both', labelsize=20)
ax_ins1.xaxis.set_major_locator(ticker.MultipleLocator(10)) 
ax_ins1.set_ylabel(r'$q_{m0}$', fontsize=20)
ax_ins1.set_xlim(0,30)

# Inset 2
ax_ins2.bar(data_2['W_num'][:n_points], data_2['q_num'][:n_points], width=0.4, 
            color='green', alpha=0.6, edgecolor='black', linewidth=0.3)
markerline, stemlines, baseline = ax_ins2.stem(
    data_2['W_th'][:n_points], data_2['q_th'][:n_points], 
    linefmt='r-', markerfmt='ro', basefmt=" "
)
plt.setp(stemlines, linewidth=0.8); plt.setp(markerline, markersize=2); plt.setp(baseline, visible=False)
ax_ins2.axhline(0, color='black', linewidth=0.5)
ax_ins2.set_xlabel(r'$W$', fontsize=20, labelpad=6)
ax_ins2.tick_params(axis='both', labelsize=20)
ax_ins2.xaxis.set_major_locator(ticker.MultipleLocator(20)) 
ax_ins2.set_ylabel(r'$q_{m2}$', fontsize=20)

# --- BOX DI TESTO ---

center_x = (pos_x1 + (pos_x2 + inset_w)) / 2
text_y = pos_y - 0.15

# Box Media Lavoro
text_str = (f"$\\langle W \\rangle$="
            f"{W_avg_theory:.2f}"
            )
ax.text(center_x, text_y, text_str, transform=ax.transAxes,
        fontsize=20, ha='center', va='top',
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9))

# NUOVO BOX: Nonpositivity Functional
# Posizionato sotto il primo box (offset di 0.12 sull'asse Y relativo)
text_str_N = (f"$\\mathcal{{N}}$="
              f"{nonpositivity_val:.4f}")

ax.text(center_x, text_y - 0.12, text_str_N, transform=ax.transAxes,
        fontsize=20, ha='center', va='top',
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9)) # Sfondo leggermente rosato

plt.tight_layout()
plt.show()

# Verifica Output Console
print("-" * 40)
print(f"Media Numerica:          {W_avg_num:.6f}")
print(f"Media Teorica:           {W_avg_theory:.6f}")
print(f"Nonpositivity Functional:{nonpositivity_val:.6f}")
print("-" * 40)