from pathlib import Path
from pymatgen.io.vasp import Vasprun
import solphin.spectral as spectral
import numpy as np
from os.path import join
from pymatgen.io.vasp import Vasprun
from sumo.cli.optplot import optplot
from matplotlib import pyplot as plt
import pymatgen.analysis.solar.slme as slme
import logging
logging.getLogger('matplotlib.font_manager').disabled = True

q=1.60217662E-19
kT=0.0258519975 # eV for T=300K
k=0.000086173325 #eV/K
h = 4.135667696e-15   # eV·s
c = 2.99792458e8      # m/s
hc = 1239.84193       # eV·nm

def spectrum_nm_to_photon_flux(spectrum):

    wavelength_nm = spectrum[:, 0]
    I = spectrum[:, 1]  # W m^-2 nm^-1

    Phi = (I * wavelength_nm) / hc   # photons m^-2 s^-1 nm^-1

    return wavelength_nm, Phi

def calc_dielectric(filename):

    '''Calculates the dielectric constants from a vasprun.xml
    
    Parameters:
        filename(string): filename/ path of the vasprun, typically vasprun.xml.
        
    Returns:
        eps_full(np.array): static dielectric constant (complex, contains both real and imaginary components)
        energies(np.array): energy of the incident radiation eV
        '''

    load_vasprun = Vasprun(filename)
    dielectric = load_vasprun.dielectric

    energies = np.array(dielectric[0])

    real_eps = np.array(dielectric[1])[:, [[0, 3, 5], [3, 1, 4], [5, 4, 2]]]
    imag_eps = np.array(dielectric[2])[:, [[0, 3, 5], [3, 1, 4], [5, 4, 2]]]
    eps_full = real_eps + 1j * imag_eps

    return eps_full, energies

def calc_absorption(eps_full, energies):

    '''Calculates the averages of the real and imaginary components of the refractive index, absorption, losses, real and imaginary components of the 
    static dielectric constant.
    
    Parameters:
        eps_full(np.array): static dielectric constant (complex, contains both real and imaginary components)
        energies(np.array): energy of the incident radiation eV
        
    Returns:
        data(dictionary):
        '''

    eps_eig = np.linalg.eigvals(eps_full)
    eps = np.mean(eps_eig, axis=1)

    n_complex = np.sqrt(eps + 0j)

    n_real = np.real(n_complex)
    k = np.imag(n_complex)

    alpha = (4 * np.pi * energies * np.imag(n_complex)) / (h * c) # consistent units

    loss = (-1 / eps).imag

    data = {
        "eps_real": np.real(eps),
        "eps_imag": np.imag(eps),
        "n_real": n_real,
        "n_imag": k,
        "loss": loss,
        "absorption": alpha,
    }

    return data

def print_n_real_file(data, energies, directory:Path):

    filename = 'n_real.dat'

    if directory:
            filename = join(directory, filename)

    header = "energy(eV)"

    header += " alpha"
    data = np.stack((energies, data['n_real']), axis=1)

    np.savetxt(filename, data, header=header)

def generate_n_real(filename):
     
     directory = Path(filename).parent
     
     eps_full, energies = calc_dielectric(filename)

     data = calc_absorption(eps_full, energies)

     print_n_real_file(data, energies, directory)

def plot_absorption(filename, xmin=0, xmax=6, gaussian=0.05):
    fig, ax = plt.subplots(figsize=(3,3), dpi=150)
    optplot(filenames=filename, xmin=xmin, xmax=xmax, gaussian=gaussian, plt=plt)
    plt.show()
    return

def blank_efficiency_energy(spectrum, energy, alpha_cm, thickness_cm, model="flat", n=3.5):
    """
    Compute Blank-style efficiency in ENERGY space.

    energy: eV
    alpha_cm: cm^-1
    thickness_cm: cm
    model: "flat" or "lambert"
    """

    lam_nm = spectrum[:, 0]
    I = spectrum[:, 1]  # W m^-2 nm^-1

    # convert to photon flux per energy
    lam_m = lam_nm * 1e-9
    E_J = (6.62607015e-34 * 2.99792458e8) / lam_m
    E_eV = E_J / 1.60217662e-19

    phi_lambda = I * lam_m / (6.62607015e-34 * 2.99792458e8)
    dlam_dE = (6.62607015e-34 * 2.99792458e8) / (E_J**2)
    phi_E = phi_lambda * dlam_dE

    # sort
    idx = np.argsort(E_eV)
    E_sun = E_eV[idx]
    phi_E = phi_E[idx]

    # interpolate solar spectrum onto material energy grid
    phi = np.interp(energy, E_sun, phi_E, left=0, right=0)

    # --- absorptance ---
    if model == "flat":
        A = 1 - np.exp(-alpha_cm * thickness_cm)

    elif model == "lambert":
        A = (alpha_cm * thickness_cm) / (alpha_cm * thickness_cm + 1/(2*n**2))

    A = np.clip(A, 0, 1)

    # --- efficiency (energy-weighted) ---
    num = np.trapz(A * phi * energy, energy)
    den = np.trapz(phi * energy, energy)

    eta = num / den * 100

    return eta


def make_blank_plot(spectrum, abs_file, direct_gap, indirect_gap):

    energy, alpha_cm = spectral.load_absorption(abs_file)

    # --- thickness grid ---
    thickness = np.logspace(-8, -3, 80)  # nm or cm depending on α units

    eff_flat = []
    eff_lam = []
    eff_slme = []  # optional placeholder

    for d in thickness:

        eff_flat.append(blank_efficiency_energy(spectrum, energy, alpha_cm, d, model="flat"))
        eff_lam.append(blank_efficiency_energy(spectrum, energy, alpha_cm, d, model="lambert"))

        # SLME placeholder (replace if you have full model)
        eff = slme.slme(energy, alpha_cm, direct_gap, indirect_gap, thickness=d, absorbance_in_inverse_centimeters=True)
        eff_slme.append(eff)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(6,4))

    plt.plot(thickness, eff_slme, label='SLME')
    plt.plot(thickness, eff_lam, label='Blank Lambertian')
    plt.plot(thickness, eff_flat, label='Blank Flat')

    plt.xscale('log')
    plt.margins(x=0)
    plt.ylim([0, 35])

    ax.set_aspect(0.06)

    plt.legend()
    plt.show()

    return thickness, eff_slme, eff_lam, eff_flat