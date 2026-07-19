import numpy as np
from pathlib import Path

# ============================================================
# PATCH SIGMA DIRETTAMENTE NELLA CARTELLA DOWNLOADS
# ============================================================

alpha = 0.1

N = 10**18
NE = int(0.7 * N)
NI = int(0.3 * N)

chiE = NE / (NE + NI)
chiI = NI / (NE + NI)

# Cartella dove hai davvero i file .npz
base_dir = Path(r"C:\Users\defra\Downloads\Wilson-Cowan new")

h_values = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]


def recompute_sigmas(data):
    E0 = data["E0"]
    I0 = data["I0"]

    x = data["A_x"]
    y = data["A_y"]
    z = data["A_z"]
    w = data["A_w"]

    G = -alpha * (E0 * chiE**2 + I0 * chiI**2)
    H = -alpha * (E0 * chiE**2 - I0 * chiI**2)

    denom = (x + w) * (x * w - y * z)

    sigma12 = np.full_like(x, np.nan, dtype=float)
    sigma11 = np.full_like(x, np.nan, dtype=float)
    sigma22 = np.full_like(x, np.nan, dtype=float)

    mask12 = np.isfinite(denom) & (np.abs(denom) > 1e-30)

    # FORMULA CORRETTA
    sigma12[mask12] = -(
        G[mask12] * (z[mask12] * w[mask12] + x[mask12] * y[mask12])
        - 2.0 * H[mask12] * x[mask12] * w[mask12]
    ) / denom[mask12]

    mask11 = np.isfinite(x) & (np.abs(x) > 1e-30)
    sigma11[mask11] = (G[mask11] - sigma12[mask11] * y[mask11]) / x[mask11]

    mask22 = np.isfinite(w) & (np.abs(w) > 1e-30)
    sigma22[mask22] = (G[mask22] - sigma12[mask22] * z[mask22]) / w[mask22]

    return sigma11, sigma12, sigma22


print("\n============================================================")
print("CARTELLA USATA")
print("============================================================")
print(base_dir)
print("Esiste?", base_dir.exists())

print("\nFile .npz presenti nella cartella:")
for f in sorted(base_dir.glob("*.npz")):
    print(" -", f.name)


for h in h_values:

    tag = f"h{h:.0e}"

    npz_path = base_dir / f"grid_dynamics_7030_{tag}.npz"
    out_npz = base_dir / f"grid_dynamics_7030_{tag}_sigmaFIX.npz"

    print("\n============================================================")
    print(f"h = {h:.0e}")
    print("============================================================")

    if not npz_path.exists():
        print("NON TROVATO:")
        print(npz_path)
        continue

    print("Trovato:")
    print(npz_path)

    old = np.load(npz_path)
    data = {key: old[key] for key in old.files}

    sigma11_new, sigma12_new, sigma22_new = recompute_sigmas(data)

    if "sigma12" in data:
        diff_s12 = np.nanmax(np.abs(sigma12_new - data["sigma12"]))
        print(f"max |sigma12_new - sigma12_old| = {diff_s12:.6e}")

    data["sigma11"] = sigma11_new
    data["sigma12"] = sigma12_new
    data["sigma22"] = sigma22_new

    np.savez(out_npz, **data)

    print("Salvato file corretto:")
    print(out_npz)

    if out_npz.exists():
        print("CHECK: creato correttamente.")
    else:
        print("ATTENZIONE: non creato.")


print("\n============================================================")
print("FILE CORRETTI CREATI")
print("============================================================")

for f in sorted(base_dir.glob("*sigmaFIX.npz")):
    print(" -", f.name)

print("\nFine.")