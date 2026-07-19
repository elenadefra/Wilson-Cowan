#Questo codice è uguale a "dynamics_E0I0_andfunctions.py" ma con una griglia diversa,   quella "giusta" che usa Manoj,
# e con A_EI ecc. calcolate e riportate già.


import numpy as np
from numba import njit, prange
import time
from scipy.io import savemat

# =======================
# PARAMETERS (FIXED)
# =======================
alpha = 0.1
wEE = 6.95
wII = 6.85

N = 10**18
NE = int(0.7 * N)
NI = int(0.3 * N)

chiE = NE / (NE + NI)
chiI = NI / (NE + NI)

dt = 1e-3
T = 32000
steps = int(T / dt)

# =======================
# GRID
# =======================
delta_EI_vals = np.linspace(-7.0, 7.0, round((7.0 - (-7.0)) / 0.1) + 1)   # 141 punti
delta_IE_vals = np.linspace(-6.9, 7.0, round((7.0 - (-6.9)) / 0.1) + 1)   # 140 punti

# =======================
# NUMBA FUNCTIONS
# =======================

@njit
def f_numba(S):
    if S > 0:
        return np.tanh(S)
    else:
        return 0.0

@njit
def f_prime(S):
    if S > 0:
        return 1.0 - np.tanh(S)**2
    else:
        return 0.0


@njit
def simulate_point(alpha, wEE, wEI, wIE, wII, h, NE, NI, dt, steps):

    k = NE // 1.2
    l = NI // 2

    size = steps // 2
    k_series = np.zeros(size)
    l_series = np.zeros(size)

    idx = 0

    for t in range(steps):

        E = k / NE
        I = l / NI

        SE = wEE * E - wEI * I + h
        SI = wIE * E - wII * I + h

        fE = f_numba(SE)
        fI = f_numba(SI)

        rateE = alpha * k + (NE - k) * fE
        rateI = alpha * l + (NI - l) * fI

        noiseE = np.sqrt(rateE * dt) * np.random.randn()
        noiseI = np.sqrt(rateI * dt) * np.random.randn()

        k += dt * (-alpha * k + (NE - k) * fE) + noiseE
        l += dt * (-alpha * l + (NI - l) * fI) + noiseI

        if k < 0:
            k = 0.0
        elif k > NE:
            k = NE

        if l < 0:
            l = 0.0
        elif l > NI:
            l = NI

        if t >= steps // 2:
            k_series[idx] = k
            l_series[idx] = l
            idx += 1

    return k_series, l_series


@njit
def compute_stats(k_series, l_series, NE, NI):

    n = len(k_series)

    k_mean = 0.0
    l_mean = 0.0
    for i in range(n):
        k_mean += k_series[i]
        l_mean += l_series[i]

    k_mean /= n
    l_mean /= n

    E0 = k_mean / NE
    I0 = l_mean / NI

    mean_xi_E = 0.0
    mean_xi_I = 0.0

    for i in range(n):
        mean_xi_E += (k_series[i] - k_mean)
        mean_xi_I += (l_series[i] - l_mean)

    mean_xi_E /= (n * np.sqrt(NE))
    mean_xi_I /= (n * np.sqrt(NI))

    std_xi_E = 0.0
    std_xi_I = 0.0

    for i in range(n):
        xiE = (k_series[i] - k_mean) / np.sqrt(NE)
        xiI = (l_series[i] - l_mean) / np.sqrt(NI)

        std_xi_E += (xiE - mean_xi_E)**2
        std_xi_I += (xiI - mean_xi_I)**2

    std_xi_E = np.sqrt(std_xi_E / n)
    std_xi_I = np.sqrt(std_xi_I / n)

    return E0, I0, mean_xi_E, std_xi_E, mean_xi_I, std_xi_I


@njit
def compute_analytics(E0, I0, alpha, wEE, wEI, wIE, wII, h, chiE, chiI):

    SE = wEE * E0 - wEI * I0 + h
    SI = wIE * E0 - wII * I0 + h

    fE  = f_numba(SE)
    fI  = f_numba(SI)
    fpE = f_prime(SE)
    fpI = f_prime(SI)

    # Coefficienti matrice A~ (eq. A16-A19)
    AEE = -alpha - fE + (1.0 - E0) * wEE * fpE
    AEI = -(1.0 - E0) * wEI * fpE
    AIE =  (1.0 - I0) * wIE * fpI
    AII = -alpha - fI - (1.0 - I0) * wII * fpI

    # Coefficienti matrice A in variabili (Sigma, Delta) eq. A24
    ratio_EI = (chiE / chiI)**1.5
    ratio_IE = (chiI / chiE)**1.5

    x = 0.5 * (AEE + ratio_EI * AEI + ratio_IE * AIE + AII)
    y = 0.5 * (AEE - ratio_EI * AEI + ratio_IE * AIE - AII)
    z = 0.5 * (AEE + ratio_EI * AEI - ratio_IE * AIE - AII)
    w = 0.5 * (AEE - ratio_EI * AEI - ratio_IE * AIE + AII)

    # Autovalori
    discriminant = (x - w)**2 + 4.0 * y * z

    if discriminant >= 0:
        sqrt_disc = np.sqrt(discriminant)
        lambda1_re = 0.5 * (x + w + sqrt_disc)
        lambda2_re = 0.5 * (x + w - sqrt_disc)
        lambda_re  = 0.5 * (x + w)
        lambda_im  = 0.0
        is_complex = 0.0
    else:
        lambda_re  = 0.5 * (x + w)
        lambda_im  = 0.5 * np.sqrt(-discriminant)
        lambda1_re = lambda_re
        lambda2_re = lambda_re
        is_complex = 1.0

    # Coefficienti di rumore (eq. A20-A21)
    DE = np.sqrt(alpha * E0 + (1.0 - E0) * fE)
    DI = np.sqrt(alpha * I0 + (1.0 - I0) * fI)

    G = -alpha * (E0 * chiE**2 + I0 * chiI**2)
    H = -alpha * (E0 * chiE**2 - I0 * chiI**2)

    # Covarianza sigma (eq. A32-A34)
    denom = (x + w) * (x * w - y * z)
    if abs(denom) > 1e-30:
        sigma12 = -(G * (z * w + x * y) + 2.0 * H * x * w) / denom
    else:
        sigma12 = 0.0

    if abs(x) > 1e-30:
        sigma11 = (G - sigma12 * y) / x
    else:
        sigma11 = 0.0

    if abs(w) > 1e-30:
        sigma22 = (G - sigma12 * z) / w
    else:
        sigma22 = 0.0

    # Sigma0 e Delta0
    Sigma0 = chiE * E0 + chiI * I0
    Delta0 = chiE * E0 - chiI * I0

    return (AEE, AEI, AIE, AII,
            lambda1_re, lambda2_re, lambda_re, lambda_im, is_complex,
            x, y, z, w,
            sigma11, sigma12, sigma22,
            Sigma0, Delta0)


@njit(parallel=True)
def run_grid(alpha, wEE, wII, h, NE, NI, dt, steps,
             delta_EI_vals, delta_IE_vals, chiE, chiI):

    Ne = len(delta_EI_vals)
    Ni = len(delta_IE_vals)

    E0_mat       = np.zeros((Ne, Ni))
    I0_mat       = np.zeros((Ne, Ni))
    mean_xi_E    = np.zeros((Ne, Ni))
    std_xi_E     = np.zeros((Ne, Ni))
    mean_xi_I    = np.zeros((Ne, Ni))
    std_xi_I     = np.zeros((Ne, Ni))

    AEE_mat      = np.zeros((Ne, Ni))
    AEI_mat      = np.zeros((Ne, Ni))
    AIE_mat      = np.zeros((Ne, Ni))
    AII_mat      = np.zeros((Ne, Ni))

    lambda1_mat    = np.zeros((Ne, Ni))
    lambda2_mat    = np.zeros((Ne, Ni))
    lambda_re_mat  = np.zeros((Ne, Ni))
    lambda_im_mat  = np.zeros((Ne, Ni))
    is_complex_mat = np.zeros((Ne, Ni))

    x_mat = np.zeros((Ne, Ni))
    y_mat = np.zeros((Ne, Ni))
    z_mat = np.zeros((Ne, Ni))
    w_mat = np.zeros((Ne, Ni))

    sigma11_mat = np.zeros((Ne, Ni))
    sigma12_mat = np.zeros((Ne, Ni))
    sigma22_mat = np.zeros((Ne, Ni))

    Sigma0_mat = np.zeros((Ne, Ni))
    Delta0_mat = np.zeros((Ne, Ni))

    for i in prange(Ne):
        for j in range(Ni):

            delta_EI = delta_EI_vals[i]
            delta_IE = delta_IE_vals[j]

            wEI = wII + delta_EI
            wIE = wEE + delta_IE

            k_series, l_series = simulate_point(
                alpha, wEE, wEI, wIE, wII, h, NE, NI, dt, steps
            )

            E0, I0, mE, sE, mI, sI = compute_stats(
                k_series, l_series, NE, NI
            )

            (aee, aei, aie, aii,
             l1, l2, lre, lim, isc,
             x, y, z, w,
             s11, s12, s22,
             Sig0, Del0) = compute_analytics(
                E0, I0, alpha, wEE, wEI, wIE, wII, h, chiE, chiI
            )

            E0_mat[i, j] = E0
            I0_mat[i, j] = I0
            mean_xi_E[i, j] = mE
            std_xi_E[i, j]  = sE
            mean_xi_I[i, j] = mI
            std_xi_I[i, j]  = sI

            AEE_mat[i, j] = aee
            AEI_mat[i, j] = aei
            AIE_mat[i, j] = aie
            AII_mat[i, j] = aii

            lambda1_mat[i, j]    = l1
            lambda2_mat[i, j]    = l2
            lambda_re_mat[i, j]  = lre
            lambda_im_mat[i, j]  = lim
            is_complex_mat[i, j] = isc

            x_mat[i, j] = x
            y_mat[i, j] = y
            z_mat[i, j] = z
            w_mat[i, j] = w

            sigma11_mat[i, j] = s11
            sigma12_mat[i, j] = s12
            sigma22_mat[i, j] = s22

            Sigma0_mat[i, j] = Sig0
            Delta0_mat[i, j] = Del0

    return (E0_mat, I0_mat, mean_xi_E, std_xi_E, mean_xi_I, std_xi_I,
            AEE_mat, AEI_mat, AIE_mat, AII_mat,
            lambda1_mat, lambda2_mat, lambda_re_mat, lambda_im_mat, is_complex_mat,
            x_mat, y_mat, z_mat, w_mat,
            sigma11_mat, sigma12_mat, sigma22_mat,
            Sigma0_mat, Delta0_mat)


# =======================
# RUN
# =======================

for h in [1e-9, 1e-10]: 

    tag = f"h{h:.0e}"

    print(f"Inizio simulazione con h={h}...")
    start = time.time()

    (E0_mat, I0_mat, mean_xi_E, std_xi_E, mean_xi_I, std_xi_I,
     AEE_mat, AEI_mat, AIE_mat, AII_mat,
     lambda1_mat, lambda2_mat, lambda_re_mat, lambda_im_mat, is_complex_mat,
     x_mat, y_mat, z_mat, w_mat,
     sigma11_mat, sigma12_mat, sigma22_mat,
     Sigma0_mat, Delta0_mat) = run_grid(
        alpha, wEE, wII, h, NE, NI, dt, steps,
        delta_EI_vals, delta_IE_vals, chiE, chiI
    )

    elapsed = time.time() - start
    print(f"Tempo impiegato per h={h}: {elapsed:.1f}s")

    # =======================
    # SAVE
    # =======================

    data = dict(
        delta_EI   = delta_EI_vals,
        delta_IE   = delta_IE_vals,
        E0         = E0_mat,
        I0         = I0_mat,
        mean_xi_E  = mean_xi_E,
        std_xi_E   = std_xi_E,
        mean_xi_I  = mean_xi_I,
        std_xi_I   = std_xi_I,
        A_EE       = AEE_mat,
        A_EI       = AEI_mat,
        A_IE       = AIE_mat,
        A_II       = AII_mat,
        lambda1    = lambda1_mat,
        lambda2    = lambda2_mat,
        lambda_re  = lambda_re_mat,
        lambda_im  = lambda_im_mat,
        is_complex = is_complex_mat,
        A_x        = x_mat,
        A_y        = y_mat,
        A_z        = z_mat,
        A_w        = w_mat,
        sigma11    = sigma11_mat,
        sigma12    = sigma12_mat,
        sigma22    = sigma22_mat,
        Sigma0     = Sigma0_mat,
        Delta0     = Delta0_mat,
    )

    np.savez(f"grid_dynamics_7030_{tag}.npz", **data)
    savemat(f"grid_dynamics_7030_{tag}.mat", data)

    print(f"Done. Saved to grid_dynamics_7030_{tag}.npz and grid_dynamics_7030_{tag}.mat")
