from pathlib import Path
from pymatgen.io.vasp import Vasprun
import solphin.spectral as spectral
import numpy as np
from os.path import join
import scipy.special as sc
import pandas as pd
from sumo.cli.optplot import optplot
from matplotlib import pyplot as plt
import pymatgen.analysis.solar.slme as slme
import logging
from importlib.resources import files
logging.getLogger('matplotlib.font_manager').disabled = True

q=1.60217662E-19
kT=0.0258519975 # eV for T=300K
k=0.000086173325 #eV/K
h_e=4.135667E-15 #eVs
h = 6.62607015e-34  # J·s
c=2.9979E+8 #m/s

def spectrum_nm_to_photon_energy(spectrum):

    wavelength_nm = spectrum[:, 0]
    irradiance = spectrum[:, 1]

    wavelength_m = wavelength_nm * 1e-9

    # energy in J
    E_J = h * c / wavelength_m

    # convert to eV immediately (cleaner)
    E_eV = E_J / q

    # photon flux per wavelength
    phi_lambda = irradiance * wavelength_m / (h * c)

    # correct Jacobian dλ/dE (in SI-consistent form)
    d_lambda_dE = (h * c) / (E_J**2)

    phi_E = phi_lambda * d_lambda_dE

    idx = np.argsort(E_eV)

    return phi_E[idx], E_eV[idx]

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

    alpha = (4 * np.pi * energies * np.imag(n_complex)) / (h_e * c) # consistent units

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


def blank_flat(alpha, n, length): 

    #Absorptance for flat scatterer, from Matlab script
    the_c = np.arcsin(1/n[0])
    theta = np.linspace(0.0, the_c, num=200)
    a_len = len(alpha)
    t_len = len(theta)
    toft = np.zeros(a_len*t_len)
    toft = toft.reshape(a_len, t_len)
    absorb_tr = np.zeros(a_len)

    for i, a_pt in enumerate(alpha):
        for j, t_pt in enumerate(theta):
            toft_pt = np.exp(((2*np.multiply((-a_pt), length))/np.cos(t_pt)))
            toft[i, j] = toft_pt
        a_t_1 = np.trapz((toft[i,:]*np.cos(theta)*np.sin(theta)), theta)
        a_t_2 = np.trapz((np.cos(theta)*np.sin(theta)), theta)
        absorb_tr[i] = 1 - (a_t_1/a_t_2)

    absorb = absorb_tr.conjugate()
    return(absorb)


def blank_lambert(alpha, n, length):

    #Absorptance for Lambertian scatterer coating, from Matlab script
    x = np.multiply(2,np.multiply(alpha,length))
    T = np.exp((-x)) - np.multiply(x, np.exp((-x))) + (x**2*sc.exp1(x))
    R = 1.0/(np.multiply(n,n))
    Abs = (1-T)/(1-T+(R*T))
    Abs = np.nan_to_num(Abs)
    Emi = (R*T)/(1-T+(R*T))

    return(Abs)

def blank_eta(spectrum, E, alpha, n, length, Qi, trap):
    #For given scatterer, calculates Blank et al. eta
    dE = np.mean(np.diff(E))
    if trap == 1:
        Abs = blank_flat(alpha, n, length)
    elif trap == 2:
        Abs = blank_lambert(alpha, n, length)
    
    #np divide necessary to have array divide here?
    E_J = E * q  # eV → J

    exp_term = np.exp(np.clip(E / kT, 1e-6, 700))  # prevents overflow
    phibb = (2 * E_J**2) / (h**3 * c**2) / (exp_term - 1)
    phibb = np.nan_to_num(phibb, nan=0.0, posinf=0.0, neginf=0.0)

    phi_sun, ps_E = spectrum_nm_to_photon_energy(spectrum)

    # convert energy grid → wavelength grid
    lam_E = 1240 / E

    # interpolate spectrum + model onto same wavelength grid
    phi_sun_E = np.interp(lam_E[::-1], ps_E[::-1], phi_sun[::-1], left=0, right=0)

    Abs_E = Abs
    alpha_E = alpha
    n_E = n
    phibb_E = phibb

    Abs_E = np.nan_to_num(Abs_E, nan=0.0)
    alpha_E = np.nan_to_num(alpha_E, nan=0.0)
    n_E = np.nan_to_num(n_E, nan=0.0)
    phibb_E = np.nan_to_num(phibb_E, nan=0.0)
    phi_sun_E = np.nan_to_num(phi_sun_E, nan=0.0)

    Jsc = q * np.trapz(Abs_E * phi_sun_E, lam_E)
    J0rad = q * np.trapz(Abs_E * phibb_E, lam_E)

    Rrad = 4*np.pi * np.trapz(alpha_E * (n_E**2) * phibb_E, lam_E)
    Rtotal = Rrad / Qi
    
    pe = J0rad / (q * Rtotal * length)
    J0 = q * length * Rtotal
    #Looks like this scans over only voltages between 0 and 2 V
    #need testing for band gaps>2 eV?
    V = np.linspace(0, 2, 1001)
    J = Jsc - J0 * (np.exp(V / kT) - 1)
    P = V * J
    Pmax = np.max(P)
    eta = Pmax/1000

    return(eta)

def blank_parse(folder):
    
    # Parses outputs from current directory

    abs_data = pd.read_table(f'{folder}/absorption.dat', sep=r"\s+",
                                skiprows=1, header=None)
    n_data = pd.read_table(f'{folder}/n_real.dat', sep=r"\s+",
                    skiprows=1, header=None)
    
    E_p = list(abs_data[0])
    alpha_p = list(abs_data[1])
    n_p = list(n_data[1])

    return{"E": E_p, "alpha": alpha_p, "n": n_p}

def blank_calculate(spectrum, folder):

    # spectrum must be (wavelength in nm, irradiance in W m^-2 nm^-1)

    data = blank_parse(folder)
    directory = Path(folder)

    E = np.asarray(data["E"])
    alpha = np.asarray(data["alpha"])
    n = np.asarray(data["n"])

    #Remove data for E>5eV, necessary for speed! Also done in Matlab

    E = np.asarray([o for o in E if o <= 5])
    alpha = alpha[0:(len(E))]
    n = n[0:(len(E))]

    #main, looping over lengths and Qi, outputs eta table
    length_arr = np.logspace(-8.0, -3.0, num=36)
    Qi_arr = np.logspace(0, -6, num=4)
    trap = [1, 2]

    for tr_pt in trap:
        eta_arr = np.zeros(len(length_arr)*(len(Qi_arr)+1))
        eta_arr = eta_arr.reshape(len(length_arr), (len(Qi_arr)+1))
        for k, l_pt in enumerate(length_arr):
            for l, q_pt in enumerate(Qi_arr):
                eta_max = blank_eta(spectrum, E, alpha, n, l_pt, q_pt, tr_pt)
                eta_arr[k, 0] = l_pt
                eta_arr[k, l+1] = eta_max
        if tr_pt == 1:
            head="Thickness[m] \t Eta as fraction for Flat scatterer with Qi = 1.0, 0.01, 1E-4, 1E-6"
            np.savetxt(f'{directory}/flat_eta_out', eta_arr, header=head)
        elif tr_pt == 2:
            head="Thickness[m] \t Eta as fraction for Lambertian scatterer with Qi = 1.0, 0.01, 1E-4, 1E-6"
            np.savetxt(f'{directory}/lamb_eta_out', eta_arr, header=head)

def plot_absorption(filename, xmin=0, xmax=6, gaussian=0.05):
    fig, ax = plt.subplots(figsize=(3,3), dpi=150)
    optplot(filenames=filename, xmin=xmin, xmax=xmax, gaussian=gaussian, plt=plt)
    plt.show()
    return

def calculate_slme(abs_file, direct_gap, indirect_gap):

    energy, alpha_cm = spectral.load_absorption(abs_file)

    thickness = np.logspace(-8, -3, 100, endpoint=True)
    effSlm = []

    for i in thickness:
        eff = slme.slme(energy, alpha_cm, direct_gap, indirect_gap, thickness=i, absorbance_in_inverse_centimeters=True)
        effSlm.append(eff)

    return effSlm, thickness

def blank_models_vs_thickness(energy, alpha, Phi_sun, lam_sun, thickness_array, n=3.5):

    eff_flat = []
    eff_lam = []

    for d in thickness_array:

        # FLAT MODEL
        A_flat = 1 - np.exp(-alpha * d)

        # LAMBERTIAN MODEL
        A_lam = alpha / (alpha + 1/(2*n**2*d))

        # efficiency functional (same as SLME-style integral)
        eta_flat = np.trapz(A_flat * Phi_sun, lam_sun) / np.trapz(Phi_sun, lam_sun)
        eta_lam  = np.trapz(A_lam  * Phi_sun, lam_sun) / np.trapz(Phi_sun, lam_sun)
        eff_flat.append(eta_flat)
        eff_lam.append(eta_lam)

    print(eff_flat)
    print(eff_lam)

    return np.array(eff_flat), np.array(eff_lam)