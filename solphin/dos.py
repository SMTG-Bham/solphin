import numpy as np
from pymatgen.io.vasp import Vasprun
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

# Physical constants (SI)
hbar = 1.054571817e-34  # J·s
eV_to_J = 1.602176634e-19
m0 = 9.10938356e-31  # electron mass (kg)

def dos_effective_mass(
    vasprun_path,
    carrier="electron",   # "electron" or "hole"
    energy_window=0.2,    # eV
    smooth_sigma=2,
    dos_threshold=1e-3
):
    """
    Compute DOS effective mass from vasprun.xml.

    Returns:
        m_eff / m0  (dimensionless)
    """

    vr = Vasprun(vasprun_path)
    dos = vr.complete_dos

    energies = dos.energies - vr.efermi  # shift EF = 0
    densities = sum(dos.densities.values())  # sum spins

    # --- Smooth DOS to reduce noise ---
    densities = gaussian_filter1d(densities, smooth_sigma)

    # --- Get cell volume in m^3 ---
    vol_A3 = vr.final_structure.volume
    vol_m3 = vol_A3 * 1e-30

    # Convert DOS: states / (eV·cell) → states / (J·m³)
    densities_SI = densities / (eV_to_J * vol_m3)

    # --- Find band edge ---
    if carrier == "electron":
        mask = (energies > 0) & (densities > dos_threshold)
        Ec = energies[mask][0]

        fit_mask = (energies > Ec) & (energies < Ec + energy_window)

        def model(E, A):
            return A * np.sqrt(E - Ec)

    elif carrier == "hole":
        mask = (energies < 0) & (densities > dos_threshold)
        Ev = energies[mask][-1]

        fit_mask = (energies < Ev) & (energies > Ev - energy_window)

        def model(E, A):
            return A * np.sqrt(Ev - E)

    else:
        raise ValueError("carrier must be 'electron' or 'hole'")

    E_fit = energies[fit_mask]
    DOS_fit = densities_SI[fit_mask]

    # Remove any negative/invalid values
    valid = DOS_fit > 0
    E_fit = E_fit[valid]
    DOS_fit = DOS_fit[valid]

    # --- Fit ---
    popt, _ = curve_fit(model, E_fit, DOS_fit, maxfev=10000)
    A = popt[0]

    # --- Extract effective mass ---
    prefactor = 2 * np.pi**2 * A
    m_eff = (hbar**2 / 2) * (prefactor ** (2/3))

    return m_eff / m0