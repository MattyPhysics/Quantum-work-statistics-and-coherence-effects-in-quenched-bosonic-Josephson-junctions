import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lpmv, gammaln

# =============================================================================
#  PARAMETRI FISICI (Sudden Quench BJJ -> QHO)
# =============================================================================
N = 100
K = 1.0
U_i = 0.1 * K   
U_f = 1.0 * K   

omega_i = K * np.sqrt(U_i * N / K + 1)
omega_f = K * np.sqrt(U_f * N / K + 1)

Q_param = (omega_i**2 + omega_f**2) / (2 * omega_i * omega_f)
u_bog = np.sqrt((Q_param + 1) / 2.0)
x_leg = 1.0 / u_bog
index_sign = np.sign(omega_f - omega_i)

m_cutoff = 50 

# =============================================================================
#  FUNZIONI CORE
# =============================================================================

def SqueezingMatrixElement(m, n, x, index_sign):
    if (m + n) % 2 != 0: return 0.0
    l = (m + n) // 2
    k = (m - n) // 2
    legendre_val = lpmv(abs(k), l, x)
    log_pref = 0.5 * (gammaln(min(n, m) + 1) - gammaln(max(n, m) + 1))
    phase = (-index_sign)**abs(k) if m >= n else (index_sign)**abs(k)
    return np.exp(log_pref) * np.sqrt(x) * legendre_val * phase

def E_harmonic(n, omega):
    return omega * (n + 0.5)

def calculate_TPM_work(c0_mod, omega_i, omega_f):
    c2_mod_sq = 1.0 - c0_mod**2
    factor = (omega_i**2 - omega_f**2) / (2 * omega_i)
    return factor * (2 * c2_mod_sq + 0.5)

def calculate_work_and_bound(c0_mod, phi):
    c2_mod = np.sqrt(1 - c0_mod**2)
    coeffs = {0: c0_mod, 2: c2_mod * np.exp(1j * phi)}
    W_real_total = 0.0
    Bound_total = 0.0
    
    for n in [0, 2]:
        p_n_TPM = abs(coeffs[n])**2
        E_i = E_harmonic(n, omega_i)
        for m in range(0, m_cutoff, 2):
            E_f = E_harmonic(m, omega_f)
            W_extracted = E_i - E_f 
            S_mn = SqueezingMatrixElement(m, n, x_leg, index_sign)
            
            interf_KDQ = 0.0
            for s in [0, 2]:
                rho_ns = coeffs[n] * np.conj(coeffs[s])
                interf_KDQ += rho_ns * SqueezingMatrixElement(m, s, x_leg, index_sign)
            
            q_real = S_mn * np.real(interf_KDQ)
            W_real_total += q_real * W_extracted
            
            if W_extracted > 0:
                p_joint_TPM = p_n_TPM * (S_mn**2)
                amp_m = coeffs[0] * SqueezingMatrixElement(m, 0, x_leg, index_sign) + \
                        coeffs[2] * SqueezingMatrixElement(m, 2, x_leg, index_sign)
                p_m_coherent = abs(amp_m)**2
                Bound_total += W_extracted * np.sqrt(p_joint_TPM * p_m_coherent)
                
    return W_real_total, Bound_total

# =============================================================================
#  GENERAZIONE DATI PER COMPARAZIONE PESI (c0)
# =============================================================================

c0_scan = np.linspace(0, 1, 50)
fixed_phi = 0 # Fase fissata per mostrare il boost massimo o minimo

w_kdq_list = []
w_tpm_list = []
bound_list = []

for c in c0_scan:
    w_k, b = calculate_work_and_bound(c, fixed_phi)
    w_t = calculate_TPM_work(c, omega_i, omega_f)
    w_kdq_list.append(w_k)
    w_tpm_list.append(w_t)
    bound_list.append(b)

# =============================================================================
#  PLOTTING
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# --- PLOT 1: Comparazione Magnitudini vs c0 ---
ax1.plot(c0_scan, w_kdq_list, 'b-', label=r'Lavoro KDQ ($W_{KDQ}$)', linewidth=2.5)
ax1.plot(c0_scan, w_tpm_list, 'k--', label=r'Lavoro TPM ($W_{TPM}$)', alpha=0.7)
ax1.plot(c0_scan, bound_list, 'r:', label=r'Bound Classico', linewidth=2)

ax1.fill_between(c0_scan, w_kdq_list, bound_list, where=(np.array(w_kdq_list) > np.array(bound_list)), 
                 color='green', alpha=0.2, label='Zona Coherence Boost')

ax1.set_title(f'Comparazione Lavoro vs Peso Popolazione $c_0$\n(Fase fissata $\phi={fixed_phi}$)', fontsize=13)
ax1.set_xlabel(r'Peso dello stato fondamentale $c_0$', fontsize=12)
ax1.set_ylabel('Energia Estratta [K]', fontsize=12)
ax1.legend(frameon=True)
ax1.grid(True, alpha=0.2)

# --- PLOT 2: Differenza (Violazione) vs c0 ---
# Calcoliamo la violazione per diverse fasi per dare una visione d'insieme
for p in [0, np.pi/2, np.pi]:
    v_list = []
    for c in c0_scan:
        wk, b = calculate_work_and_bound(c, p)
        v_list.append(wk - b)
    ax2.plot(c0_scan, v_list, label=r'$\phi = ' + f'{p/np.pi:.1f}' + r'\pi$')

ax2.axhline(0, color='red', linestyle='--', linewidth=1)
ax2.set_title('Violazione del Bound Classico ($W_{KDQ} - Bound$)', fontsize=13)
ax2.set_xlabel(r'Peso dello stato fondamentale $c_0$', fontsize=12)
ax2.set_ylabel('Delta Energia [K]', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()