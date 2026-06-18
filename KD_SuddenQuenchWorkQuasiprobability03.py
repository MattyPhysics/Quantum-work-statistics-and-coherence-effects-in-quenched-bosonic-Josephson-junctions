import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from scipy.special import lpmv, gammaln
import matplotlib.ticker as ticker
import matplotlib.colors as colors


def twomode_bose_hubbard_hamiltonian(N, K, U):
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

def TransitionAmplitude(m, n, x, index_sign):
    if (m + n) % 2 != 0: 
        return 0
    l = (m + n) // 2
    k = (m - n) // 2
    
    legendrevalue = lpmv(abs(k), l, x)
    
    log_pref = 0.5 * (gammaln(min(n, m) + 1) - gammaln(max(n, m) + 1))
    
    if m >= n:
        phase = (index_sign)**abs(k)
    else:
        phase = (-index_sign)**abs(k)
        
    return np.exp(log_pref) * np.sqrt(x) * legendrevalue * phase

def numerical_phases(V):
    for i in range(V.shape[1]):
        idx = np.argmax(np.abs(V[:, i]))
        phase = np.sign(V[idx, i])
        if phase != 0:
            V[:, i] *= phase
    return V

def E_th_func(k, omega):
    return omega * (k + 0.5)
##############################################################################################################################à

N = 100
K = 1.0
U_i =  0.1* K
U_f = 0.0*K

# Diagonalizzazione
H_i = twomode_bose_hubbard_hamiltonian(N, K, U_i)
H_f = twomode_bose_hubbard_hamiltonian(N, K, U_f)
E_i, V_i = eigh(H_i)
E_f, V_f = eigh(H_f)

V_i = numerical_phases(V_i)
V_f = numerical_phases(V_f)

# Stato iniziale 
c_coeffs = np.zeros(N + 1, dtype=complex)
c_coeffs[0] = 0.303030
c_coeffs[2] = np.sqrt(1-c_coeffs[0]**2)
rho_0 = np.outer(c_coeffs, np.conj(c_coeffs))

omega_i = K * np.sqrt(U_i * N / K + 1) 
omega_f = K * np.sqrt(U_f * N / K + 1) 
index_sign = np.sign(omega_i - omega_f)

r=1/2*np.log(omega_f/omega_i)
x_leg = 1.0 / np.cosh(np.abs(r))

TransitionAmplitudeNum = V_f.T @ V_i 

# KD
q_num_matrix = np.zeros((N + 1, N + 1), dtype=complex)
q_theory_matrix = np.zeros((N + 1, N + 1), dtype=complex)

#TPM
p_num_matrix = np.zeros((N + 1, N + 1))
p_theory_matrix = np.zeros((N + 1, N + 1))

for n in [0, 2]:
    for m in range(0, N+ 1, 2):
        interf_num = sum(rho_0[n, s] * np.conj(TransitionAmplitudeNum[m, s]) for s in [0, 2])
        q_num_matrix[m, n] = TransitionAmplitudeNum[m, n] * interf_num
        
        interf_theory = sum(rho_0[n, s] * TransitionAmplitude(m, s, x_leg, index_sign) for s in [0, 2])
        q_theory_matrix[m, n] = TransitionAmplitude(m, n, x_leg, index_sign) * interf_theory

for n in [0, 2]:
    p_n = np.abs(rho_0[n,n])
    for m in range(0, N+ 1, 2):
        prob_trans_num = np.abs(TransitionAmplitudeNum[m, n])**2
        p_num_matrix[m, n] = p_n * prob_trans_num
        lambda_mn_th = TransitionAmplitude(m, n, x_leg, index_sign)
        prob_trans_th = np.abs(lambda_mn_th)**2
        p_theory_matrix[m, n] = p_n * prob_trans_th

# Nonpositivity
non_positivity_num = np.sum(np.abs(q_num_matrix)) - 1.0
non_positivity_theory = np.sum(np.abs(q_theory_matrix)) - 1.0
imaginary_sum=np.sum(q_theory_matrix.imag)
###############################################################################################################################
# Mean/Variance KD

# Numerics
W_avg_num = 0
W_sq_avg_num = 0

for n in [0, 2]:
    for m in range(N + 1):
        work_val = E_f[m] - E_i[n]
        val_q = q_num_matrix[m, n].real 
        
        W_avg_num += val_q * work_val
        W_sq_avg_num += val_q * (work_val**2)

Var_num = W_sq_avg_num - W_avg_num**2

# Mean/Variance KD

# Numerics

W_avg_numTPM = 0
for n in [0, 2]:
    for m in range(N + 1):
        work_val = E_f[m] - E_i[n]
        W_avg_numTPM += p_num_matrix[m, n] * work_val

W_sq_avg_numTPM = 0
for n in [0, 2]:
    for m in range(N + 1):
        work_val = E_f[m] - E_i[n]
        W_sq_avg_numTPM += p_num_matrix[m, n] * (work_val**2)

Var_numTPM = W_sq_avg_numTPM - W_avg_numTPM**2
################################################################################################################à

print(f'Media lavoro KD',W_avg_num)
print(f'Media lavoro TPM',W_avg_numTPM)
print(f'Varianza lavoro KD',Var_num)
print(f'Varianza lavoro TPM',Var_numTPM)
#################################################################################################à
#  PLOT QUASIPROBABILITY
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
    for key in d: d[key] = np.array(d[key])

fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')

# --- Plot Principale ---
ax.bar(data_0['W_num'], data_0['q_num'], width=0.3, color='blue', alpha=0.5, 
       edgecolor='blue', linewidth=0.5, label=r'Bose-Hubbard $q_{0,2m}$')
ax.bar(data_2['W_num'], data_2['q_num'], width=0.3, color='green', alpha=0.5, 
       edgecolor='green', linewidth=0.5, label=r'Bose-Hubbard $q_{2,2m}$')

markerline0, stemlines0, baseline0 = ax.stem(data_0['W_th'], data_0['q_th'], 
                                             linefmt='r-', markerfmt='ro', basefmt=" ", label='Analytical (QHO)')
markerline2, stemlines2, baseline2 = ax.stem(data_2['W_th'], data_2['q_th'], 
                                             linefmt='r-', markerfmt='ro', basefmt=" ")

plt.setp([markerline0, markerline2], markersize=4)
plt.setp([stemlines0, stemlines2], linewidth=1.0, alpha=0.6)
plt.setp([baseline0, baseline2], visible=False)

ax.axhline(0, color='black', linewidth=1.2)
ax.set_xlabel(r'$w/K$', fontsize=25)
ax.set_ylabel(r'$P(w)$', fontsize=25)
ax.set_xlim(-10, 10)
ax.set_ylim(-0.1, 0.4)
ax.grid(True, alpha=0.15, linestyle='--')
ax.legend(loc='upper left', fontsize=18)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.xaxis.set_major_locator(ticker.MultipleLocator(2.0))
ax.tick_params(axis='both', labelsize=22)

# Insets 
inset_w = 0.40; inset_h = 0.35; pos_y = 0.70    
ax_ins1 = ax.inset_axes([0.42, pos_y, 0.245, 0.27]) 
ax_ins2 = ax.inset_axes([0.75, pos_y, 0.245, 0.27])


n_points = 10 
# Inset 1
ax_ins1.bar(data_0['W_num'][:n_points], data_0['q_num'][:n_points], width=0.4, 
            color='blue', alpha=0.6, edgecolor='black', linewidth=0.3)
#ax_ins1.stem(data_0['W_th'][:n_points], data_0['q_th'][:n_points], 
             #linefmt='r-', markerfmt='ro',basefmt=" ")
mal1, sl1, bl1= ax_ins1.stem(
    data_0['W_th'][:n_points], 
    data_0['q_th'][:n_points], 
    linefmt='r-', 
    markerfmt='ro', 
    basefmt=" "
)

# 2. Modifica la proprietà del markerline
plt.setp(mal1, markersize=2)
ax_ins1.axhline(0, color='black', linewidth=0.5)
ax_ins1.set_xlabel(r'$w$', fontsize=16, labelpad=6) 
ax_ins1.tick_params(axis='both', labelsize=15)
ax_ins1.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax_ins1.yaxis.set_major_locator(ticker.MultipleLocator(0.2)) 
ax_ins1.set_ylabel(r'$q_{0,2m}$', fontsize=16)
ax_ins1.set_xlim(-2, 10)
ax_ins1.set_ylim(-0.1, 0.2)

# Inset 2
ax_ins2.bar(data_2['W_num'][:n_points], data_2['q_num'][:n_points], width=0.6, 
            color='green', alpha=0.6, edgecolor='black', linewidth=0.3)
#ax_ins2.stem(data_2['W_th'][:n_points], data_2['q_th'][:n_points], 
             #linefmt='r-', markerfmt='ro', basefmt=" ")
mal2, sl2, bl2= ax_ins2.stem(
    data_2['W_th'][:n_points], 
    data_2['q_th'][:n_points], 
    linefmt='r-', 
    markerfmt='ro', 
    basefmt=" "
)

# 2. Modifica la proprietà del markerline
plt.setp(mal2, markersize=2)
ax_ins2.axhline(0, color='black', linewidth=0.5)
ax_ins2.set_xlabel(r'$w$', fontsize=16, labelpad=6)
ax_ins2.tick_params(axis='both', labelsize=15)
ax_ins2.xaxis.set_major_locator(ticker.MultipleLocator(4))
ax_ins2.yaxis.set_major_locator(ticker.MultipleLocator(0.2)) 
ax_ins2.set_ylabel(r'$q_{2,2m}$', fontsize=16)
ax_ins2.set_xlim(-10, 10)
ax_ins2.set_ylim(-0.1, 0.3)

#  Box 
"""
center_x = (0.6+ (0.8+ inset_w)) / 2
text_y = pos_y - 0.46


text_str = (
    r"$ \mathcal{W} = %.2f$" % (-W_avg_num, ) + "\n" +
    r"$Re[(\Delta \mathcal{W})^2]= %.2f$" % (Var_num, ) + "\n" +
    r"$\mathcal{N} = %.4f$" % (non_positivity_theory, )
)

ax.text(center_x, text_y, text_str, transform=ax.transAxes,
        fontsize=22, ha='center', va='top',
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.5))

"""

plt.tight_layout()
plt.savefig('KD_superpos.pdf', format='pdf', bbox_inches='tight')
plt.show()

####################################################################################
# Plot TPM

m_max_plot = 40 

data_0 = {'W_num': [], 'p_num': [], 'W_th': [], 'p_th': []}
data_2 = {'W_num': [], 'p_num': [], 'W_th': [], 'p_th': []}

for n, dataset in zip([0, 2], [data_0, data_2]):
    for m in range(0, m_max_plot, 2):
        w_n = E_f[m] - E_i[n]
        val_p = p_num_matrix[m, n]
        if val_p > 1e-10:
            dataset['W_num'].append(w_n)
            dataset['p_num'].append(val_p)
            
        w_t = E_th_func(m, omega_f) - E_th_func(n, omega_i)
        val_pt = p_theory_matrix[m, n]
        if val_pt > 1e-10:
            dataset['W_th'].append(w_t)
            dataset['p_th'].append(val_pt)

for d in [data_0, data_2]:
    for key in d:
        d[key] = np.array(d[key])

# =============================================================================
#  PLOT PRINCIPALE CON INSET E MEDIA LAVORO
# =============================================================================
# =============================================================================
#  PLOT TPM (Uniformato al primo)
# =============================================================================

fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')

# --- 1. Plot Principale ---
# Usiamo width=0.3 come nel primo plot per coerenza
ax.bar(data_0['W_num'], data_0['p_num'], width=0.3, color='blue', alpha=0.5, 
       edgecolor='blue', linewidth=0.5, label=r'Bose-Hubbard $q_{0,2m}$')
ax.bar(data_2['W_num'], data_2['p_num'], width=0.3, color='green', alpha=0.5, 
       edgecolor='green', linewidth=0.5, label=r'Bose-Hubbard $q_{2,2m}$')

markerline0, stemlines0, baseline0 = ax.stem(data_0['W_th'], data_0['p_th'], 
                                             linefmt='r-', markerfmt='ro', basefmt=" ", label='Analytical (QHO)')
markerline2, stemlines2, baseline2 = ax.stem(data_2['W_th'], data_2['p_th'], 
                                             linefmt='r-', markerfmt='ro', basefmt=" ")

plt.setp([markerline0, markerline2], markersize=4)
plt.setp([stemlines0, stemlines2], linewidth=1.0, alpha=0.6)
plt.setp([baseline0, baseline2], visible=False)

# Configurazione Assi (uguale al primo plot)

ax.axhline(0, color='black', linewidth=1.2)
ax.set_xlabel(r'$w/K$', fontsize=25)
ax.set_ylabel(r'$P(w)$', fontsize=25)
ax.set_xlim(-10, 10)
ax.set_ylim(0.0, 0.4)
ax.grid(True, alpha=0.15, linestyle='--')
ax.legend(loc='upper left', fontsize=18)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.xaxis.set_major_locator(ticker.MultipleLocator(2.0))
ax.tick_params(axis='both', labelsize=22)

# --- 2. Insets (posizionati e dimensionati come il primo) ---

inset_w = 0.40; inset_h = 0.35; pos_y = 0.70    
ax_ins1 = ax.inset_axes([0.42, pos_y, 0.245, 0.27]) 
ax_ins2 = ax.inset_axes([0.75, pos_y, 0.245, 0.27])


# Inset 1
ax_ins1.bar(data_0['W_num'][:n_points], data_0['p_num'][:n_points], width=0.4, 
            color='blue', alpha=0.6, edgecolor='black', linewidth=0.2)
mal3, sl3, bl3 = ax_ins1.stem(
    data_0['W_th'][:n_points], data_0['p_th'][:n_points], # Corretto 'p_th'
    linefmt='r-', markerfmt='ro', basefmt=" "
)

# 2. Modifica la proprietà del markerline
plt.setp(mal3, markersize=2)
ax_ins1.axhline(0, color='black', linewidth=0.5)
ax_ins1.set_xlim(-2, 10)
ax_ins1.set_ylabel(r'$q_{0,2m}$', fontsize=20)
ax_ins1.set_xlabel(r'$w$', fontsize=16, labelpad=6) 
ax_ins1.tick_params(axis='both', labelsize=16)
ax_ins1.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax_ins1.yaxis.set_major_locator(ticker.MultipleLocator(0.2)) 
ax_ins1.set_ylabel(r'$q_{0,2m}$', fontsize=16)
ax_ins1.set_xlim(-2, 10)
ax_ins1.set_ylim(0, 0.1)

# Inset 2
ax_ins2.bar(data_2['W_num'][:n_points], data_2['p_num'][:n_points], width=0.6, 
            color='green', alpha=0.6, edgecolor='black', linewidth=0.3)
mal4, sl4, bl4 = ax_ins2.stem( # Corretto ax_ins2
    data_2['W_th'][:n_points], data_2['p_th'][:n_points], # Corretto 'data_2' e 'p_th'
    linefmt='r-', markerfmt='ro', basefmt=" "
)

# 2. Modifica la proprietà del markerline
plt.setp(mal4, markersize=2)
ax_ins2.axhline(0, color='black', linewidth=0.5)
ax_ins2.set_xlim(-10, 10) # Range X identico
ax_ins2.set_ylabel(r'$q_{2,2m}$', fontsize=16)
ax_ins2.set_xlabel(r'$w$', fontsize=16, labelpad=6) 
ax_ins2.tick_params(axis='both', labelsize=16)
ax_ins2.xaxis.set_major_locator(ticker.MultipleLocator(4))
ax_ins2.yaxis.set_major_locator(ticker.MultipleLocator(0.2)) 
ax_ins2.set_ylim(0, 0.3)
"""
center_x = (0.45+ (0.8+ inset_w)) / 2
text_y = pos_y - 0.45


text_str = (
    r"$\mathcal{W} = %.2f$" % (-W_avg_numTPM, ) + "\n" +
    r"$Re[(\Delta \mathcal{W})^2]= %.2f$" % (Var_numTPM, ) + "\n" +
    r"$\mathcal{N} = %.0f$" % (0, )
)
ax.text(center_x, text_y, text_str, transform=ax.transAxes,
        fontsize=22, ha='center', va='top',
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.5))

"""



plt.tight_layout()
plt.savefig('KD_diagonal.pdf', format='pdf', bbox_inches='tight')
plt.show()

print(f'Valore della non-positività:',non_positivity_num)