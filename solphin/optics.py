from pymatgen.io.vasp import Vasprun
import numpy as np
from os.path import join
import scipy.special as sc
import pandas as pd
from sumo.cli.optplot import optplot
from matplotlib import pyplot as plt
import logging
logging.getLogger('matplotlib.font_manager').disabled = True

q=1.60217662E-19
kT=0.0258519975 # eV for T=300K
k=0.000086173325 #eV/K
h=4.135667E-15 #eVs
c=2.9979E+8 #m/s

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

    # take sqrt of eps matrix; if eps = V S V^-1; then eps^1/2 = V S^{1/2} V^-1;
    eigvals, eigvecs = np.linalg.eig(eps_full)

    # fancy einsum to calculate V S^{1/2} V^-1 at every energy
    n = np.einsum("ijk,ik,ikl->ijl", eigvecs, np.sqrt(eigvals), np.linalg.inv(eigvecs))

    # calculate optical absorption
    alpha = n.imag * energies[:, None, None] * 4 * np.pi / 1.23984212e-4

    # Invert epsilon to obtain energy-loss function
    loss = -np.linalg.inv(eps_full).imag

    eps = np.linalg.eigvals(eps_full).mean(axis=1)
    n = np.linalg.eigvals(n).mean(axis=1)
    loss = np.linalg.eigvalsh(loss).mean(axis=1)
    alpha = np.linalg.eigvalsh(alpha).mean(axis=1)

    data = {
        "eps_real": eps.real,
        "eps_imag": eps.imag,
        "n_real": n.real,
        "n_imag": n.imag,
        "loss": loss,
        "absorption": alpha,
    }

    return data

def print_n_real_file(data, energies):

    filename = 'n_real.dat'
    directory = 'dos'

    if directory:
            filename = join(directory, filename)

    header = "energy(eV)"

    header += " alpha"
    data = np.stack((energies, data['n_real']), axis=1)

    np.savetxt(filename, data, header=header)

def generate_n_real(filename):
     
     eps_full, energies = calc_dielectric(filename)

     data = calc_absorption(eps_full, energies)

     print_n_real_file(data, energies)


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
    dE = E[1]-E[0]
    if trap == 1:
        Abs = blank_flat(alpha, n, length)
    elif trap == 2:
        Abs = blank_lambert(alpha, n, length)
    
    #np divide necessary to have array divide here?
    phibb = 2*np.divide(np.divide(np.multiply(E,E),((h**3)*(c**2))),(np.exp(E/kT)-1))
    phibb = np.nan_to_num(phibb) # NaNs to 0, as in Matlab

    ps_E = spectrum[:, 0]
    phi_sun = spectrum[:, 1]
    
    # works for all E values above 0.03? won't extrapolate for 0
    # values < gap shouldn't be relevant?
    phisun = 10000*np.interp(E, ps_E, phi_sun)
        
    Jsc = q*np.sum(Abs*phisun)*dE
    J0rad = q*np.sum(Abs*phibb)*dE
    
    Rrad = 4*np.pi*np.sum(alpha*(n**2)*phibb)*dE
    Rnrad = (Rrad-Qi*Rrad)/Qi
    
    pe = J0rad/(q*Rrad*length)
    J0 = q*length*(Rnrad + pe*Rrad)
    #Looks like this scans over only voltages between 0 and 2 V
    #need testing for band gaps>2 eV?
    V = np.linspace(0, 2, 1001)
    Pmax = np.max(V*(Jsc - J0*(np.exp(V/kT)-1)))
    eta = Pmax/1000

    return(eta)

def blank_parse(folder):
    
    # Parses outputs from current directory

    abs_data = pd.read_table(f'{folder}/absorption.dat', delim_whitespace=True,
                                skiprows=1, header=None)
    n_data = pd.read_table(f'{folder}/n_real.dat', delim_whitespace=True,
                    skiprows=1, header=None)
    
    E_p = list(abs_data[0])
    alpha_p = list(abs_data[1])
    n_p = list(n_data[1])

    return{"E": E_p, "alpha": alpha_p, "n": n_p}

def blank_calculate(spectrum, folder):

    data = blank_parse(folder)

    E = np.asarray(data["E"])
    alpha = np.asarray(data["alpha"])
    alpha = np.multiply(alpha, 100)
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
            np.savetxt('flat_eta_out', eta_arr, header=head)
        elif tr_pt == 2:
            head="Thickness[m] \t Eta as fraction for Lambertian scatterer with Qi = 1.0, 0.01, 1E-4, 1E-6"
            np.savetxt('lamb_eta_out', eta_arr, header=head)

def plot_absorption(filename, xmin=0, xmax=6, gaussian=0.05):
    fig, ax = plt.subplots(figsize=(3,3), dpi=150)
    optplot(filenames=filename, xmin=xmin, xmax=xmax, gaussian=gaussian, plt=plt)
    plt.show()
    return