import numpy as np
from scipy.special import lpmv, gammaln

def TransitionAmplitude(m, n, x, index_sign):
    if (m + n) % 2 != 0: return 0
    l = (m + n) // 2
    k = (m - n) // 2
    
    # lpmv(m, l, x) calcola P_l^m(x)
    legendrevalue = lpmv(abs(k), l, x)
    
    log_pref = 0.5 * (gammaln(min(n, m) + 1) - gammaln(max(n, m) + 1))
    
    if m >= n:
        phase = (index_sign)**abs(k)
    else:
        phase = (-index_sign)**abs(k)
        
    return np.exp(log_pref) * np.sqrt(x) * legendrevalue * phase

def check_paper_relations():
    # Parametri test
    r = 0.5
    x = 1.0 / np.cosh(r)
    # index_sign nel paper è sgn(omega_i - omega_f)
    sgn = -1 
    
    print(f"{'s':<5} | {'Eq 37 (Num)':<15} | {'Eq 37 (Ana)':<15} | {'Eq 35 (Num)':<15} | {'Eq 35 (Ana)':<15}")
    print("-" * 75)
    
    for s in range(1, 4):
        # Lambda calcolate tramite la tua funzione
        L_00 = TransitionAmplitude(0, 0, x, sgn)
        L_2s0 = TransitionAmplitude(2*s, 0, x, sgn)
        L_2s2 = TransitionAmplitude(2*s, 2, x, sgn)
        
        # Rapporti numerici
        f_num = L_2s0 / L_00
        g_num = L_2s2 / L_2s0
        
        # Formule analitiche del paper
        # Nota: La 37 ha un fattore sqrt(x) o simili che deve bilanciare 
        # la definizione della tua funzione.
        f_ana = -sgn * (2*s + 1) * np.sqrt(x*(1 - x**2)/2)  # Verifica fattore 1/2
        g_ana = sgn * (1 - x**2 * (1 + 2*s)) / np.sqrt((2 * (1 - x**2)))
        
        print(f"{s:<5} | {f_num:<15.6f} | {f_ana:<15.6f} | {g_num:<15.6f} | {g_ana:<15.6f}")

check_paper_relations()