from pathlib import Path
from pymatgen.io.vasp import Vasprun
import solphin.spectral as spectral
from solphin.db_fom import load_spectrum
import numpy as np
from os.path import join
from sumo.cli.optplot import optplot
from matplotlib import pyplot as plt
import pymatgen.analysis.solar.slme as slme_mod
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
import os
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from scipy import constants

h_eV = 4.135667696e-15   # eV s
c    = 2.99792458e8       # m/s
hc_eV_nm = 1239.84193    # eV nm

_c   = constants.c
_h   = constants.h
_h_e = constants.h / constants.e
_k   = constants.k
_k_e = constants.k / constants.e
_e   = constants.e
_T   = 293.15


def calc_dielectric(filename):
    load_vasprun = Vasprun(filename)
    dielectric   = load_vasprun.dielectric
    energies     = np.array(dielectric[0])
    real_eps     = np.array(dielectric[1])[:, [[0,3,5],[3,1,4],[5,4,2]]]
    imag_eps     = np.array(dielectric[2])[:, [[0,3,5],[3,1,4],[5,4,2]]]
    eps_full     = real_eps + 1j * imag_eps
    return eps_full, energies


def calc_absorption(eps_full, energies):

    eps_eig = np.linalg.eigvals(eps_full)

    # Scalar averaged dielectric (for eps outputs and loss function)
    eps = np.mean(eps_eig, axis=1)

    # Per-eigenvalue refractive index, then average (sumo-consistent)
    n_eig     = np.sqrt(eps_eig + 0j)
    n_complex = np.mean(n_eig, axis=1)

    n_real = np.real(n_complex)
    k      = np.imag(n_complex)

    alpha = (4 * np.pi * energies * k) / (h_eV * c)   # m-1
    loss  = (-1 / eps).imag

    return {
        "eps_real":   np.real(eps),
        "eps_imag":   np.imag(eps),
        "n_real":     n_real,
        "n_imag":     k,
        "loss":       loss,
        "absorption": alpha,
    }


def print_n_real_file(data, energies, directory: Path):
    filename = "n_real.dat"
    if directory:
        filename = join(directory, filename)
    out = np.stack((energies, data["n_real"]), axis=1)
    np.savetxt(filename, out, header="energy(eV) n_real")


def generate_n_real(filename):
    directory          = Path(filename).parent
    eps_full, energies = calc_dielectric(filename)
    data               = calc_absorption(eps_full, energies)
    print_n_real_file(data, energies, directory)


def plot_absorption(filename, xmin=0, xmax=6, gaussian=0.05):
    fig, ax = plt.subplots(figsize=(3, 3), dpi=150)
    optplot(filenames=filename, xmin=xmin, xmax=xmax, gaussian=gaussian, plt=plt)
    plt.show()

def spectrum_select(spectrum_type):

    use_slme = (spectrum_type == "AM1.5")

    # --- solar / illuminant spectrum in wavelength space ---
    if use_slme:
        am15_path = os.path.join(os.path.dirname(slme_mod.__file__), "am1.5G.dat")
        sol_wl, sol_irr = np.loadtxt(am15_path, usecols=[0, 1],
                                      unpack=True, skiprows=2)  # nm, W m-2 nm-1
    else:
        spectrum = load_spectrum(spectrum_type)
        sol_wl   = spectrum[:, 0]   # nm
        sol_irr  = spectrum[:, 1]   # W m-2 nm-1

    return sol_wl, sol_irr, use_slme

def convert_spec(sol_wl, sol_irr):

    sol_wl_m = sol_wl * 1e-9 # Convert wavelength to meters
    sol_phot_flux = sol_irr * (sol_wl_m / (_h * _c))  # photons m-2 s-1 nm-1

    return sol_wl_m, sol_phot_flux

def make_blank_plot(abs_file, n_real_file, direct_gap, indirect_gap,
                    spectrum_type="AM1.5", Qi=1.0, n=3.5, thickness_range=None):
    
    # Setup the spectrum and convert to units
    sol_wl, sol_irr, use_slme = spectrum_select(spectrum_type)
    sol_wl_m, sol_phot_flux = convert_spec(sol_wl, sol_irr)
       
    # Calculate incident power
    power_in = simpson(sol_irr, x=sol_wl)          # W m-2

    # blackbody photon flux in wavelength space [photons m-2 s-1 m-1]
    bb_irr     = (2*_h*_c**2 / sol_wl_m**5) / (np.exp(_h*_c/(sol_wl_m*_k*_T)) - 1.0)
    bb_phot_wl = bb_irr * (sol_wl_m / (_h * _c))

    # blackbody photon flux in energy space [photons m-2 s-1 eV-1] for pe integral
    def _bb_per_eV(E_eV):
        E_J = E_eV * _e
        exp = np.clip(E_eV / ((_k / _e) * _T), 0, 700)
        return (2*E_J**2) / (_h**3*_c**2) / (np.exp(exp) - 1.0 + 1e-300) * _e

    # --- absorption and n_real data (on the same energy grid) ---
    energy_abs, alpha_cm = spectral.load_absorption(abs_file)
    alpha_m = alpha_cm * 1e2   # cm-1 → m-1

    nr_dat  = np.loadtxt(n_real_file, comments='#')
    n_real  = np.interp(energy_abs, nr_dat[:, 0], nr_dat[:, 1])

    phi_bb_E  = _bb_per_eV(energy_abs)

    # pe denominator: ∫n²(E)·α(E)·φ_BB(E) dE  — independent of thickness
    denom_int = simpson(n_real**2 * alpha_m * phi_bb_E, x=energy_abs)

    # --- interpolate alpha onto solar wavelength grid (pymatgen style) ---
    wl_alpha   = ((_c * _h_e) / (energy_abs + 1e-8)) * 1e9   # nm
    alpha_func = interp1d(wl_alpha, alpha_m, kind='cubic',
                          fill_value=(alpha_m[0], alpha_m[-1]),
                          bounds_error=False)

    wl_gap_nm    = (_c * _h_e / direct_gap) * 1e9
    alpha_on_sol = np.zeros(len(sol_wl))
    for i, wl in enumerate(sol_wl):
        if wl < wl_gap_nm:
            alpha_on_sol[i] = alpha_func(wl)

    # fr factor (same as SLME, for indirect gap materials)
    delta = direct_gap - indirect_gap
    fr    = np.exp(-delta / (_k_e * _T))

    # --- efficiency with full pe/Qi correction (Blank et al. eqs. 4-6) ---
    def _eta_d(d, A_sol, A_E):
        numer_int = simpson(A_E * phi_bb_E, x=energy_abs)
        pe  = min(numer_int / (4 * d * denom_int), 1.0)

        # External luminescence efficiency (Blank eq. after eq. 6)
        Qe  = (pe * Qi) / (1.0 + (pe - 1.0) * Qi)

        # J0_rad (standard detailed balance, wavelength space)
        J0_rad = _e * np.pi * simpson(bb_phot_wl * A_sol, x=sol_wl_m)
        J0     = J0_rad / Qe   # total saturation current

        # # fr correction for indirect gap (same as SLME)
        # J0 = J0 / fr

        Jsc = _e * simpson(sol_phot_flux * A_sol, x=sol_wl)
        if J0 <= 0 or Jsc <= 0:
            return 0.0

        def Jfn(V): return Jsc - J0 * (np.exp(_e*V / (_k*_T)) - 1.0)
        def Pfn(V): return Jfn(V) * V
        tv = 0.0; vs = 0.001
        while Pfn(tv + vs) > Pfn(tv):
            tv += vs
        return Pfn(tv) / power_in * 100.0    

    eff_flat, eff_lam, eff_slme = thickness_calc(thickness_range, alpha_on_sol, alpha_m, _eta_d, use_slme, n, energy_abs, alpha_cm, direct_gap, indirect_gap)
            
    plot_blank(use_slme, thickness_range, eff_slme, eff_lam, eff_flat)

def thickness_calc(thickness_range, alpha_on_sol, alpha_m, _eta_d, use_slme, n, energy_abs, alpha_cm, direct_gap, indirect_gap):
        
        if thickness_range is None:
            thickness_range = np.logspace(-8, -3, 80)   # m
        
        eff_flat = []; eff_lam = []; eff_slme = []

        for d in thickness_range:
            # absorptance on solar wavelength grid (for Jsc, J0_rad)
            A_flat_sol = np.clip(1.0 - np.exp(-2.0 * alpha_on_sol * d), 0.0, 1.0)
            A_lamb_sol = np.clip(1.0 - 1.0/(1.0 + 4.0*n**2 * alpha_on_sol * d), 0.0, 1.0)
            # absorptance on energy grid (for pe numerator integral)
            A_flat_E   = np.clip(1.0 - np.exp(-2.0 * alpha_m * d), 0.0, 1.0)
            A_lamb_E   = np.clip(1.0 - 1.0/(1.0 + 4.0*n**2 * alpha_m * d), 0.0, 1.0)

            eff_flat.append(_eta_d(d, A_flat_sol, A_flat_E))
            eff_lam.append(_eta_d(d,  A_lamb_sol, A_lamb_E))

            if use_slme:
                eff_slme.append(slme_mod.slme(
                    energy_abs, alpha_cm, direct_gap, indirect_gap,
                    thickness=d, absorbance_in_inverse_centimeters=True))
            
        return eff_flat, eff_lam, eff_slme
            
def plot_blank(use_slme, thickness_range, eff_slme, eff_lam, eff_flat):

    fig, ax = plt.subplots(figsize=(6, 4))
    if use_slme:
        ax.plot(thickness_range, eff_slme, label="SLME")
    ax.plot(thickness_range, eff_lam,  label="Blank Lambertian")
    ax.plot(thickness_range, eff_flat, label="Blank Flat")
    ax.set_xscale("log")
    ax.set_xlabel("Film Thickness / m", labelpad=5)
    ax.set_ylabel(r"Max PV Efficiency $(\eta_\mathrm{Max})$ / %")
    ax.set_ylim([0, 35])
    ax.margins(x=0)
    ax.set_aspect(0.06)
    ax.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Backwards-compatible spectrum utilities
# ---------------------------------------------------------------------------

def spectrum_nm_to_photon_flux(spectrum):
    """Photon flux in wavelength space [photons m-2 s-1 nm-1]."""
    wavelength_nm = spectrum[:, 0]
    I             = spectrum[:, 1]
    Phi           = (I * wavelength_nm) / hc_eV_nm
    return wavelength_nm, Phi


def spectrum_nm_to_photon_energy(spectrum):
    """Convert irradiance [W m-2 nm-1] to photon flux [photons m-2 s-1 eV-1].

    Uses the correct Jacobian I(E) = I(λ)|dλ/dE| then phi(E) = I(E)/E_J.
    """
    from scipy import constants
    _h = constants.h; _c = constants.c; _e = constants.e

    lam_nm = spectrum[:, 0]
    I_lam  = spectrum[:, 1]
    lam_m  = lam_nm * 1e-9
    E_J    = _h * _c / lam_m
    E_eV   = E_J / _e
    I_E    = I_lam * hc_eV_nm / E_eV**2   # W m-2 eV-1
    phi_E  = I_E / E_J                     # photons m-2 s-1 eV-1
    idx    = np.argsort(E_eV)
    return phi_E[idx], E_eV[idx]
