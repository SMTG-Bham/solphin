from pymatgen.io.vasp import Vasprun
import numpy as np
from os.path import join


def calc_dielectric(filename):

    load_vasprun = Vasprun(filename)
    dielectric = load_vasprun.dielectric

    energies = np.array(dielectric[0])

    real_eps = np.array(dielectric[1])[:, [[0, 3, 5], [3, 1, 4], [5, 4, 2]]]
    imag_eps = np.array(dielectric[2])[:, [[0, 3, 5], [3, 1, 4], [5, 4, 2]]]
    eps_full = real_eps + 1j * imag_eps

    return eps_full, energies

def calc_absorption(eps_full, energies):

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