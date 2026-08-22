import logging
from pathlib import Path

import numpy as np
import pymatgen.analysis.solar.slme as slme_mod
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from numpy.typing import NDArray
from pymatgen.io.vasp import Vasprun
from scipy import constants as sc
from scipy.integrate import simpson
from scipy.interpolate import interp1d

import solphin.spectral as spectral
from solphin.db_fom import load_spectrum

logging.getLogger('matplotlib.font_manager').disabled = True

hc_eV_nm = 1239.84193  # eV nm

_c = sc.c
_h = sc.h
_h_e = sc.h / sc.e
_k = sc.k
_e = sc.e
_T = 293.15


def calc_dielectric(
        filename: str | Path
) -> tuple[float, NDArray, NDArray, NDArray, NDArray]:
    """
    Calculates the dielectric constants from a vasprun.xml
    
    Parameters:
        filename(string): filename/ path of the vasprun, typically vasprun.xml.
        
    Returns:
        eps_full(np.array): static dielectric constant (complex, contains both real and imaginary components)
        energies(np.array): energy of the incident radiation eV
    """

    load_vasprun = Vasprun(filename)
    dielectric = load_vasprun.dielectric

    energies = np.array(dielectric[0])
    eps_real = np.array(dielectric[1])[:, [[0, 3, 5], [3, 1, 4], [5, 4, 2]]]
    eps_imag = np.array(dielectric[2])[:, [[0, 3, 5], [3, 1, 4], [5, 4, 2]]]
    eps_full = eps_real + 1j * eps_imag

    eps_inf = np.mean(eps_real[0].diagonal())
    eps_inf_tensor = eps_real[0]

    return eps_inf, eps_inf_tensor, eps_full, eps_imag, energies


def calc_absorption(eps_full: NDArray, energies: NDArray) -> dict[str, NDArray]:
    """
    Calculates optical properties from the complex dielectric tensor.

    Parameters:
        eps_full(np.array): frequency-dependent dielectric tensor 
            with complex components. Expected shape is (N, 3, 3),
            where N is the number of energy points.
        energies(np.array): incident photon energies in eV corresponding
            to each dielectric tensor entry.

    Returns:
        dict: Dictionary containing derived optical properties:
            eps_real(np.array): real part of the averaged dielectric function.
            eps_imag(np.array): imaginary part of the averaged dielectric function.
            n_real(np.array): real part of the complex refractive index.
            n_imag(np.array): imaginary part of the complex refractive index (extinction coefficient, k).
            loss(np.array): energy loss function Im(-1/ε).
            absorption(np.array): absorption coefficient in m^-1.
    """

    eps_eig = np.linalg.eigvals(eps_full)

    # Scalar averaged dielectric (for eps outputs and loss function)
    eps = np.mean(eps_eig, axis=1)

    # Per-eigenvalue refractive index, then average (sumo-consistent)
    n_eig = np.sqrt(eps_eig + 0j)
    n_complex = np.mean(n_eig, axis=1)

    n_real = np.real(n_complex)
    k = np.imag(n_complex)

    alpha = (4 * np.pi * energies * k) / (_h_e * _c)  # m-1
    loss = (-1 / eps).imag

    return {
        "eps_real": np.real(eps),
        "eps_imag": np.imag(eps),
        "n_real": n_real,
        "n_imag": k,
        "loss": loss,
        "absorption": alpha,
    }


def print_n_real_file(
        data: dict[str, NDArray], energies: NDArray, directory: str | Path
) -> None:
    """
    Writes the real part of the refractive index to a data file.

    Parameters:
        data(dict): dictionary containing calculated optical properties.
            Must contain the key "n_real".
        energies(np.array): incident photon energies in eV corresponding
            to the optical property data.
        directory(Path): output directory where the file will be written.

    Returns:
        None
    """
    out_path = Path(directory) / "n_real.dat" if directory else Path("n_real.dat")
    out = np.stack((energies, data["n_real"]), axis=1)
    np.savetxt(out_path, out, header="energy(eV) n_real")


def print_absorption_file(
        data: dict[str, NDArray], energies: NDArray, directory: str | Path
) -> None:
    """
    Writes the absorption coefficient in cm^-1.
    """
    out_path = Path(directory) / "absorption.dat" if directory else Path("absorption.dat")

    # calc_absorption returns alpha in m^-1
    alpha_cm = data["absorption"] / 100.0

    out = np.column_stack((energies, alpha_cm))

    np.savetxt(
        out_path,
        out,
        header="energy(eV) absorption(cm^-1)"
    )


def generate_absorption(optics_directory: str | Path) -> None:
    """
    Generates and writes the real part of the refractive index from a VASP optics calculation.

    Parameters:
        optics_directory(string or Path): directory containing the
            vasprun.xml file and where the output file will be written.

    Returns:
        None
    """

    filename = f'{optics_directory}/vasprun.xml'

    _, _, eps_full, _, energies = calc_dielectric(filename)
    data = calc_absorption(eps_full, energies)
    print_absorption_file(data, energies, optics_directory)


def generate_n_real(optics_directory: str | Path) -> None:
    """
    Generates and writes the real part of the refractive index from a VASP optics calculation.

    Parameters:
        optics_directory(string or Path): directory containing the
            vasprun.xml file and where the output file will be written.

    Returns:
        None
    """

    filename = f'{optics_directory}/vasprun.xml'

    _, _, eps_full, _, energies = calc_dielectric(filename)
    data = calc_absorption(eps_full, energies)
    print_n_real_file(data, energies, optics_directory)


def plot_absorption(
        optics_directory: str | Path, xmax: float = 4, xmin: float = 0, save: bool = False
) -> None:
    """
    Plots the optical absorption spectrum from a VASP optics calculation.

    Parameters:
        optics_directory(string or Path): directory containing the
            vasprun.xml file.
        xmin(float): minimum energy value (eV) shown on the x-axis.
            Default is 0.
        xmax(float): maximum energy value (eV) shown on the x-axis.
            Default is 6.
        save(boolean): allows the ability to save the figure as png in the current directory.
            Default is False.

    Returns:
        None
    """

    filename = f'{optics_directory}/vasprun.xml'
    eps_inf, eps_inf_tensor, eps_full, eps_imag, energies = calc_dielectric(filename)
    data = calc_absorption(eps_full, energies)

    plt.figure(figsize=(3, 5))
    absorption = data["absorption"] / 1e7

    plt.plot(
        energies,
        absorption,
        linewidth=1.8,
        color='#1f77b4')

    plt.gca().xaxis.set_major_formatter(
        FormatStrFormatter('%.1f')
    )

    plt.xlabel("Photon energy (eV)", fontsize=16)
    plt.ylabel(
        r"Absorption coefficient (10$^{5}$ cm$^{-1}$)",
        fontsize=16,
    )

    plt.subplots_adjust(
        left=0.15,
        right=0.95,
        bottom=0.12,
        top=0.95,
    )

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.ylim(0, 1)
    plt.xlim(xmin, xmax)

    if save:
        plt.savefig("absorption.png", dpi=700)

    plt.show()


def _spectrum_select(spectrum_type: str) -> tuple[NDArray, NDArray, bool]:
    """
    Selects and loads a solar or illuminant spectrum.

    Parameters:
        spectrum_type(string): identifier for the spectrum to load.
            If set to "AM1.5", the standard AM1.5G solar spectrum
            included with the SLME package is used. Otherwise,
            the spectrum is loaded using load_spectrum().

    Returns:
        sol_wl(np.array): wavelength values of the spectrum in nm.
        sol_irr(np.array): spectral irradiance values in
            W m^-2 nm^-1.
        use_slme(bool): True if the built-in AM1.5G spectrum was used,
            otherwise False.
    """

    use_slme = (spectrum_type == "AM1.5")

    # --- solar / illuminant spectrum in wavelength space ---
    if use_slme:
        am15_path = Path(slme_mod.__file__).parent / "am1.5G.dat"
        sol_wl, sol_irr = np.loadtxt(am15_path, usecols=[0, 1],
                                     unpack=True, skiprows=2)  # nm, W m-2 nm-1
    else:
        spectrum = load_spectrum(spectrum_type)
        sol_wl = spectrum[:, 0]  # nm
        sol_irr = spectrum[:, 1]  # W m-2 nm-1

    return sol_wl, sol_irr, use_slme


def _convert_spec(sol_wl: NDArray, sol_irr: NDArray) -> tuple[NDArray, NDArray]:
    """
    Converts a spectral irradiance distribution into photon flux.

    Parameters:
        sol_wl(np.array): wavelength values of the spectrum in nm.
        sol_irr(np.array): spectral irradiance values in
            W m^-2 nm^-1.

    Returns:
        sol_wl_m(np.array): wavelength values converted to meters.
        sol_phot_flux(np.array): photon flux in
            photons m^-2 s^-1 nm^-1.
    """

    sol_wl_m = sol_wl * 1e-9  # Convert wavelength to meters
    sol_phot_flux = sol_irr * (sol_wl_m / (_h * _c))  # photons m-2 s-1 nm-1

    return sol_wl_m, sol_phot_flux


def _calc_incident_power(sol_irr: NDArray, sol_wl: NDArray) -> float:
    """
    Calculates the total incident power from a spectral irradiance distribution.

    Parameters:
        sol_irr(np.array): spectral irradiance values in
            W m^-2 nm^-1.
        sol_wl(np.array): wavelength values corresponding to the
            irradiance spectrum in nm.

    Returns:
        power_in(float): total incident power density in W m^-2,
            obtained by integrating the spectrum over wavelength.
    """

    power_in = simpson(sol_irr, x=sol_wl)  # W m-2

    return power_in


def _bb_per_eV(E_eV: NDArray) -> NDArray:
    """
    Computes the blackbody photon flux spectrum in energy space.

    This function evaluates the photon flux of a blackbody radiator
    at temperature _T, expressed in units of photons m^-2 s^-1 eV^-1.

    Parameters:
        E_eV(np.array): photon energies in electron volts (eV).

    Returns:
        np.array: blackbody photon flux in
            photons m^-2 s^-1 eV^-1.
    """

    # blackbody photon flux in energy space [photons m-2 s-1 eV-1] for pe integral

    E_J = E_eV * _e
    exp = np.clip(E_eV / ((_k / _e) * _T), 0, 700)
    return (2 * E_J ** 2) / (_h ** 3 * _c ** 2) / (np.exp(exp) - 1.0 + 1e-300) * _e


def _bb_per_wl(sol_wl_m: NDArray) -> NDArray:
    """
    Computes the blackbody photon flux spectrum in wavelength space.

    This function evaluates the spectral photon flux of a blackbody
    radiator at temperature _T in units of photons m^-2 s^-1 m^-1.

    Parameters:
        sol_wl_m(np.array): wavelength values in meters.

    Returns:
        bb_phot_wl(np.array): blackbody photon flux in
            photons m^-2 s^-1 m^-1.
    """

    # blackbody photon flux in wavelength space [photons m-2 s-1 m-1]
    bb_irr = (2 * _h * _c ** 2 / sol_wl_m ** 5) / (np.exp(_h * _c / (sol_wl_m * _k * _T)) - 1.0)
    bb_phot_wl = bb_irr * (sol_wl_m / (_h * _c))

    return bb_phot_wl


def _n_real_abs_fit(
        abs_file: str | Path, n_real_file: str | Path
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Loads absorption data and interpolates the real refractive index onto the same energy grid.

    This function reads absorption coefficient data and real refractive index data,
    ensures they share a common energy grid, and returns both in consistent units
    for further optical analysis.

    Parameters:
        abs_file(string or Path): file containing absorption data.
            Expected format: energy (eV), absorption coefficient (cm^-1).
        n_real_file(string or Path): file containing real refractive index data.
            Expected format: energy (eV), n_real.

    Returns:
        energy_abs(np.array): energy grid in eV from the absorption dataset.
        alpha_cm(np.array): absorption coefficient in cm^-1.
        alpha_m(np.array): absorption coefficient converted to m^-1.
        n_real(np.array): real refractive index interpolated onto the
            absorption energy grid.
    """

    # --- absorption and n_real data (on the same energy grid) ---
    energy_abs, alpha_cm = spectral._load_absorption(abs_file)
    alpha_m = alpha_cm * 1e2  # cm-1 → m-1

    nr_dat = np.loadtxt(n_real_file, comments='#')
    n_real = np.interp(energy_abs, nr_dat[:, 0], nr_dat[:, 1])

    return energy_abs, alpha_cm, alpha_m, n_real


def _interpolate_a(
        energy_abs: NDArray, alpha_m: NDArray, direct_gap: float, sol_wl: NDArray
) -> NDArray:
    """
    Interpolates the absorption coefficient onto a solar wavelength grid,
    with a cutoff below the direct band gap.

    This function converts absorption data from energy space into wavelength
    space, interpolates it onto the solar spectrum grid, and enforces a
    band-gap cutoff (no absorption for wavelengths corresponding to energies
    below the direct gap).

    Parameters:
        energy_abs(np.array): energy grid in eV corresponding to absorption data.
        alpha_m(np.array): absorption coefficient in m^-1.
        direct_gap(float): direct band gap energy in eV.
        sol_wl(np.array): solar spectrum wavelength grid in nm.

    Returns:
        alpha_on_sol(np.array): absorption coefficient interpolated onto the
            solar wavelength grid, with values set to zero below the band-gap
            cutoff wavelength.
    """

    # --- interpolate alpha onto solar wavelength grid (pymatgen style) ---
    wl_alpha = ((_c * _h_e) / (energy_abs + 1e-8)) * 1e9  # nm
    alpha_func = interp1d(wl_alpha, alpha_m, kind='cubic',
                          fill_value=(alpha_m[0], alpha_m[-1]),
                          bounds_error=False)

    wl_gap_nm = (_c * _h_e / direct_gap) * 1e9
    alpha_on_sol = np.zeros(len(sol_wl))
    for i, wl in enumerate(sol_wl):
        if wl < wl_gap_nm:
            alpha_on_sol[i] = alpha_func(wl)

    return alpha_on_sol


def make_blank_plot(
        optics_directory: str | Path,
        direct_gap: float,
        indirect_gap: float,
        spectrum_type: str = "AM1.5",
        Qi: float = 1.0,
        n: float = 3.5,
        thickness_range: NDArray | None = None,
        save: bool = False,
) -> None:
    """
    Generates a blank efficiency plot for optical absorption analysis as a function of thickness.

    This function loads absorption and refractive index data, selects a solar spectrum,
    computes photon fluxes, and evaluates efficiency models across a range of thicknesses.
    It then produces a comparison plot for different efficiency assumptions.

    Parameters:
        optics_directory(string or Path): directory containing optical data files:
            - absorption.dat
            - n_real.dat
        direct_gap(float): direct band gap energy in eV.
        indirect_gap(float): indirect band gap energy in eV.
        spectrum_type(string): type of solar spectrum to use (default is "AM1.5").
        Qi(float): internal quantum efficiency factor (default is 1.0).
        n(float): refractive index used in model calculations (default is 3.5).
        thickness_range(np.array or None): array of thickness values to evaluate.
            If None, a default range is used.

    Returns:
        None
    """

    abs_file = f'{optics_directory}/absorption.dat'
    n_real_file = f'{optics_directory}/n_real.dat'

    # Setup the spectrum and convert to units
    sol_wl, sol_irr, use_slme = _spectrum_select(spectrum_type)
    sol_wl_m, sol_phot_flux = _convert_spec(sol_wl, sol_irr)

    # Calculate indicent power

    power_in = _calc_incident_power(sol_irr, sol_wl)

    bb_phot_wl = _bb_per_wl(sol_wl_m)

    energy_abs, alpha_cm, alpha_m, n_real = _n_real_abs_fit(abs_file, n_real_file)

    eff_flat, eff_lam, eff_slme, thickness_range = _thickness_calc(thickness_range, alpha_m, use_slme, n,
                                                                   energy_abs, alpha_cm, direct_gap, indirect_gap,
                                                                   n_real, bb_phot_wl,
                                                                   sol_wl_m, sol_phot_flux, sol_wl, Qi, power_in)

    linestyle = "--" if np.isclose(direct_gap, indirect_gap) else "-"

    plot_blank(use_slme, thickness_range, eff_slme, eff_lam, eff_flat, linestyle, save)


def power_efficiency(
        A_E: NDArray, energy_abs: NDArray, n_real: NDArray, alpha_m: NDArray, d: float
) -> float:
    """
    Computes the power conversion efficiency using a spectral absorption model.

    This function evaluates the efficiency based on a photon flux-weighted
    absorption spectrum and a thickness-dependent normalization factor, following
    a detailed balance / spectral efficiency framework.

    Parameters:
        A_E(np.array): energy-dependent absorption function or absorptance
            evaluated on the energy grid.
        energy_abs(np.array): energy grid in eV.
        n_real(np.array): real refractive index evaluated on the same energy grid.
        alpha_m(np.array): absorption coefficient in m^-1.
        d(float): material thickness in meters.

    Returns:
        pe(float): calculated power efficiency (dimensionless, capped at 1.0).
    """

    phi_bb_E = _bb_per_eV(energy_abs)

    # pe denominator: ∫n²(E)·α(E)·φ_BB(E) dE  — independent of thickness
    denom_int = simpson(n_real ** 2 * alpha_m * phi_bb_E, x=energy_abs)

    # --- efficiency with full pe/Qi correction (Blank et al. eqs. 4-6) ---
    numer_int = simpson(A_E * phi_bb_E, x=energy_abs)
    pe = min(numer_int / (4 * d * denom_int), 1.0)

    return pe


def _eta_d(
        d: float,
        A_sol: NDArray,
        A_E: NDArray,
        energy_abs: NDArray,
        n_real: NDArray,
        alpha_m: NDArray,
        bb_phot_wl: NDArray,
        sol_wl_m: NDArray,
        sol_phot_flux: NDArray,
        sol_wl: NDArray,
        Qi: float,
        power_in: float,
) -> float:
    """
    Calculates the power conversion efficiency for a given thickness using detailed-balance optics.

    This function evaluates thickness-dependent efficiency by combining optical absorption,
    radiative recombination limits, and external luminescence efficiency within a
    detailed-balance framework.

    Parameters:
        d(float): material thickness in meters.
        A_sol(np.array): wavelength-dependent absorptance on the solar spectrum grid.
        A_E(np.array): energy-dependent absorptance on the energy grid.
        energy_abs(np.array): energy grid in eV.
        n_real(np.array): real refractive index evaluated on the energy grid.
        alpha_m(np.array): absorption coefficient in m^-1.
        bb_phot_wl(np.array): blackbody photon flux in wavelength space.
        sol_wl_m(np.array): solar wavelength grid in meters.
        sol_phot_flux(np.array): solar photon flux in wavelength space.
        sol_wl(np.array): solar wavelength grid in nm.
        Qi(float): internal quantum efficiency factor.
        power_in(float): incident solar power density in W m^-2.

    Returns:
        float: power conversion efficiency (%) at the optimal operating point.
    """

    pe = power_efficiency(A_E, energy_abs, n_real, alpha_m, d)

    # External luminescence efficiency (Blank eq. after eq. 6)
    Qe = (pe * Qi) / (1.0 + (pe - 1.0) * Qi)

    # J0_rad (standard detailed balance, wavelength space)
    J0_rad = _e * np.pi * simpson(bb_phot_wl * A_sol, x=sol_wl_m)
    J0 = J0_rad / Qe  # total saturation current

    Jsc = _e * simpson(sol_phot_flux * A_sol, x=sol_wl)
    if J0 <= 0 or Jsc <= 0:
        return 0.0

    def Jfn(V: float) -> float:
        return Jsc - J0 * (np.exp(_e * V / (_k * _T)) - 1.0)

    def Pfn(V: float) -> float:
        return Jfn(V) * V

    tv = 0.0;
    vs = 0.001
    while Pfn(tv + vs) > Pfn(tv):
        tv += vs
    return Pfn(tv) / power_in * 100.0


def _thickness_calc(
        thickness_range: NDArray | None,
        alpha_m: NDArray,
        use_slme: bool,
        n: float,
        energy_abs: NDArray,
        alpha_cm: NDArray,
        direct_gap: float,
        indirect_gap: float,
        n_real: NDArray,
        bb_phot_wl: NDArray,
        sol_wl_m: NDArray,
        sol_phot_flux: NDArray,
        sol_wl: NDArray,
        Qi: float,
        power_in: float,
) -> tuple[list[float], list[float], list[float], NDArray]:
    """
Computes thickness-dependent power conversion efficiencies using different optical models.

This function evaluates efficiency as a function of material thickness using two
absorption models (Beer-Lambert and optical interference approximation), and optionally
compares against the SLME model if available.

Parameters:
    thickness_range(np.array or None): array of thickness values in meters.
        If None, a default logarithmic range from 1e-8 to 1e-3 m is used.
    alpha_m(np.array): absorption coefficient in m^-1 on the energy grid.
    use_slme(bool): whether to also compute SLME efficiency.
    n(float): refractive index used in optical model.
    energy_abs(np.array): energy grid in eV.
    alpha_cm(np.array): absorption coefficient in cm^-1 (used for SLME).
    direct_gap(float): direct band gap energy in eV.
    indirect_gap(float): indirect band gap energy in eV.
    n_real(np.array): real refractive index on the energy grid.
    bb_phot_wl(np.array): blackbody photon flux in wavelength space.
    sol_wl_m(np.array): solar wavelength grid in meters.
    sol_phot_flux(np.array): solar photon flux in wavelength space.
    sol_wl(np.array): solar wavelength grid in nm.
    Qi(float): internal quantum efficiency factor.
    power_in(float): incident solar power density in W m^-2.

Returns:
    eff_flat(list): efficiency values using exponential Beer–Lambert absorption model.
    eff_lam(list): efficiency values using optical interference-enhanced model.
    eff_slme(list): SLME efficiency values (empty if use_slme is False).
    thickness_range(np.array): thickness values used for evaluation.
"""
    alpha_on_sol = _interpolate_a(energy_abs, alpha_m, direct_gap, sol_wl)

    if thickness_range is None:
        thickness_range = np.logspace(-8, -3, 80)  # m

    eff_flat = [];
    eff_lam = [];
    eff_slme = []

    for d in thickness_range:
        # absorptance on solar wavelength grid (for Jsc, J0_rad)
        A_flat_sol = np.clip(1.0 - np.exp(-2.0 * alpha_on_sol * d), 0.0, 1.0)
        A_lamb_sol = np.clip(1.0 - 1.0 / (1.0 + 4.0 * n ** 2 * alpha_on_sol * d), 0.0, 1.0)
        # absorptance on energy grid (for pe numerator integral)
        A_flat_E = np.clip(1.0 - np.exp(-2.0 * alpha_m * d), 0.0, 1.0)
        A_lamb_E = np.clip(1.0 - 1.0 / (1.0 + 4.0 * n ** 2 * alpha_m * d), 0.0, 1.0)

        eff_flat.append(
            _eta_d(d, A_flat_sol, A_flat_E, energy_abs, n_real, alpha_m, bb_phot_wl, sol_wl_m, sol_phot_flux, sol_wl,
                   Qi, power_in))
        eff_lam.append(
            _eta_d(d, A_lamb_sol, A_lamb_E, energy_abs, n_real, alpha_m, bb_phot_wl, sol_wl_m, sol_phot_flux, sol_wl,
                   Qi, power_in))

        if use_slme:
            eff_slme.append(slme_mod.slme(
                energy_abs, alpha_cm, direct_gap, indirect_gap,
                thickness=d, absorbance_in_inverse_centimeters=True))

    return eff_flat, eff_lam, eff_slme, thickness_range


def plot_blank(
        use_slme: bool,
        thickness_range: NDArray,
        eff_slme: list[float],
        eff_lam: list[float],
        eff_flat: list[float],
        linestyle: str,
        save: bool,
) -> None:
    """
    Plots thickness-dependent maximum photovoltaic efficiency for different optical models.

    This function visualizes and compares efficiency curves obtained from:
    - SLME model (if enabled),
    - Blank Lambertian optical model,
    - Flat (Beer-Lambert) optical model.

    Parameters:
        use_slme(bool): whether SLME results are included and plotted.
        thickness_range(np.array): array of film thickness values in meters.
        eff_slme(list or np.array): SLME efficiency values (may be empty if not used).
        eff_lam(list or np.array): efficiency values from Lambertian optical model.
        eff_flat(list or np.array): efficiency values from flat Beer-Lambert model.

    Returns:
        None
    """

    fig, ax = plt.subplots(figsize=(8, 4))

    if use_slme:
        ax.plot(thickness_range, eff_slme, color='blue', label="SLME")
    ax.plot(thickness_range, eff_lam, color='green', label="Blank Lambertian")
    ax.plot(thickness_range, eff_flat, color='orange', label="Blank Flat", linestyle=linestyle)
    ax.set_xscale("log")
    ax.set_xlabel("Film Thickness / m", labelpad=5)
    ax.set_ylabel(r"Max PV Efficiency $(\eta_\mathrm{Max})$ / %")
    ax.set_ylim((0, 35))
    ax.margins(x=0)
    ax.legend()
    plt.tight_layout()
    if save:
        plt.savefig("slme.png", dpi=700)
    plt.show()
