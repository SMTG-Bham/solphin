"""Optical properties from VASP or CASTEP dielectric data: absorption, refractive index and SLME.

VASP supplies the frequency-dependent dielectric tensor directly in
``vasprun.xml``. CASTEP does not write one itself: its spectral task produces
optical matrix elements that OptaDOS turns into a ``<seed>_epsilon.dat``
file, which is what the ``code="castep"`` paths here parse - in either the
default polycrystalline geometry or the full ``optics_geom : tensor`` form.
"""

import re
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

hc_eV_nm = 1239.84193  # eV nm

_c = sc.c
_h = sc.h
_h_e = sc.h / sc.e
_k = sc.k
_e = sc.e
_T = 293.15


def _read_optados_epsilon(filename: str | Path) -> tuple[NDArray, NDArray]:
    """Parse an OptaDOS ``<seed>_epsilon.dat`` file into a dielectric tensor.

    The parser is structure-driven rather than header-driven, since OptaDOS
    header wording varies between versions: comment lines (blank, ``#`` or
    ``!``) separate blocks of numeric rows, each row being energy in eV, then
    the real and imaginary dielectric components. One data block is the
    polycrystalline geometry, expanded onto an isotropic tensor; six blocks
    are the ``optics_geom : tensor`` components in OptaDOS order (xx, yy,
    zz, xy, xz, yz), filled in symmetrically. Explicit ``Component: i j``
    comments, when present before every block, override the positional
    order.

    Parameters
    ----------
    filename : str or Path
        Path of the OptaDOS epsilon output file.

    Returns
    -------
    energies : numpy.ndarray
        Energies of the incident radiation in eV, shape (N,).
    eps_full : numpy.ndarray
        Complex frequency-dependent dielectric tensor, shape (N, 3, 3).

    Raises
    ------
    ValueError
        If a data line cannot be parsed, the tensor blocks disagree on the
        energy grid, or the block count is neither 1 nor 6.
    """
    path = Path(filename)

    blocks: list[NDArray] = []
    block_comments: list[str] = []
    current_rows: list[list[float]] = []
    pending_comments: list[str] = []

    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped[0] in "#!":
            if current_rows:
                blocks.append(np.array(current_rows))
                block_comments.append(" ".join(pending_comments))
                current_rows = []
                pending_comments = []
            if stripped:
                pending_comments.append(stripped)
            continue

        tokens = stripped.split()
        try:
            row = [float(token) for token in tokens[:3]]
        except ValueError as exc:
            raise ValueError(f"Unparseable line in {path}: {raw!r}") from exc
        if len(tokens) < 3:
            raise ValueError(
                f"Expected at least 3 columns (energy, real, imag) in {path},"
                f" got {raw!r}"
            )
        current_rows.append(row)

    if current_rows:
        blocks.append(np.array(current_rows))
        block_comments.append(" ".join(pending_comments))

    if len(blocks) == 1:
        data = blocks[0]
        energies = data[:, 0]
        eps_scalar = data[:, 1] + 1j * data[:, 2]
        eps_full = eps_scalar[:, None, None] * np.eye(3)
        return energies, eps_full

    if len(blocks) == 6:
        energies = blocks[0][:, 0]
        for block in blocks[1:]:
            if not np.allclose(block[:, 0], energies):
                raise ValueError(
                    f"Tensor blocks in {path} disagree on the energy grid."
                )

        order = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
        component_re = re.compile(r"component\s*:?\s*(\d)\s+(\d)", re.IGNORECASE)
        matches = [component_re.search(comment) for comment in block_comments]
        if all(matches):
            order = [
                (int(match.group(1)) - 1, int(match.group(2)) - 1)
                for match in matches
                if match is not None
            ]

        eps_full = np.zeros((len(energies), 3, 3), dtype=complex)
        for (i, j), block in zip(order, blocks):
            component = block[:, 1] + 1j * block[:, 2]
            eps_full[:, i, j] = component
            eps_full[:, j, i] = component
        return energies, eps_full

    raise ValueError(
        f"Expected 1 (polycrystalline) or 6 (tensor) data blocks in {path},"
        f" found {len(blocks)}."
    )


def _calc_dielectric_castep(
        filename: str | Path
) -> tuple[float, NDArray, NDArray, NDArray, NDArray]:
    """Calculate the dielectric constants from an OptaDOS epsilon file.

    Parameters
    ----------
    filename : str or Path
        Path of the OptaDOS ``<seed>_epsilon.dat`` output file.

    Returns
    -------
    tuple
        The same five values as ``calc_dielectric``: the static dielectric
        constant is taken from the lowest-energy row, matching the row-0
        convention of the VASP path.
    """
    energies, eps_full = _read_optados_epsilon(filename)

    eps_inf_tensor = np.real(eps_full[0])
    eps_inf = float(np.mean(eps_inf_tensor.diagonal()))
    eps_imag = np.imag(eps_full)

    return eps_inf, eps_inf_tensor, eps_full, eps_imag, energies


def _find_epsilon_file(optics_directory: str | Path, seedname: str | None) -> Path:
    """Locate the OptaDOS epsilon file inside a CASTEP optics directory.

    Parameters
    ----------
    optics_directory : str or Path
        Directory containing the OptaDOS output.
    seedname : str or None
        CASTEP seed. If given, ``<seedname>_epsilon.dat`` is required; if
        None, the directory must hold exactly one ``*_epsilon.dat`` file.

    Returns
    -------
    Path
        Path of the epsilon file.

    Raises
    ------
    FileNotFoundError
        If the named or globbed epsilon file does not exist.
    ValueError
        If no seedname was given and several epsilon files match.
    """
    directory = Path(optics_directory)

    if seedname is not None:
        path = directory / f"{seedname}_epsilon.dat"
        if not path.is_file():
            raise FileNotFoundError(f"No OptaDOS epsilon file at {path}")
        return path

    candidates = sorted(directory.glob("*_epsilon.dat"))
    if not candidates:
        raise FileNotFoundError(f"No *_epsilon.dat file found in {directory}")
    if len(candidates) > 1:
        names = ", ".join(candidate.name for candidate in candidates)
        raise ValueError(
            f"Several OptaDOS epsilon files in {directory}: {names};"
            " pass seedname to choose one."
        )
    return candidates[0]


def _resolve_optics_file(
        optics_directory: str | Path, code: str, seedname: str | None
) -> str | Path:
    """Resolve the dielectric data file for a directory-based optics function.

    Parameters
    ----------
    optics_directory : str or Path
        Directory containing the optics calculation output.
    code : str
        Which first-principles code produced the data, ``"vasp"`` or
        ``"castep"``.
    seedname : str or None
        CASTEP seed used to disambiguate the epsilon file; ignored for VASP.

    Returns
    -------
    str or Path
        Path of the file ``calc_dielectric`` should parse.

    Raises
    ------
    ValueError
        If ``code`` is not ``"vasp"`` or ``"castep"``.
    """
    if code == "vasp":
        return f"{optics_directory}/vasprun.xml"
    if code == "castep":
        return _find_epsilon_file(optics_directory, seedname)
    raise ValueError(f"Unsupported code {code!r}; expected 'vasp' or 'castep'.")


def calc_dielectric(
        filename: str | Path, code: str = "vasp"
) -> tuple[float, NDArray, NDArray, NDArray, NDArray]:
    """Calculate the dielectric constants from a first-principles output file.

    Parameters
    ----------
    filename : str or Path
        Path of the dielectric data file: ``vasprun.xml`` for VASP, or the
        OptaDOS ``<seed>_epsilon.dat`` for CASTEP.
    code : str, optional
        Which code produced the file, ``"vasp"`` or ``"castep"``. Default is
        ``"vasp"``.

    Returns
    -------
    eps_inf : float
        Static dielectric constant, averaged over the tensor diagonal.
    eps_inf_tensor : numpy.ndarray
        Static dielectric tensor (real part at zero energy), shape (3, 3).
    eps_full : numpy.ndarray
        Complex frequency-dependent dielectric tensor, shape (N, 3, 3).
    eps_imag : numpy.ndarray
        Imaginary part of the dielectric tensor, shape (N, 3, 3).
    energies : numpy.ndarray
        Energies of the incident radiation in eV.

    Raises
    ------
    ValueError
        If ``code`` is not ``"vasp"`` or ``"castep"``.
    """
    if code == "castep":
        return _calc_dielectric_castep(filename)
    if code != "vasp":
        raise ValueError(f"Unsupported code {code!r}; expected 'vasp' or 'castep'.")

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
    """Calculate optical properties from the complex dielectric tensor.

    Parameters
    ----------
    eps_full : numpy.ndarray
        Frequency-dependent dielectric tensor with complex components,
        shape (N, 3, 3) where N is the number of energy points.
    energies : numpy.ndarray
        Incident photon energies in eV for each dielectric tensor entry.

    Returns
    -------
    dict of str to numpy.ndarray
        Derived optical properties, keyed by ``"eps_real"`` and ``"eps_imag"``
        (averaged dielectric function), ``"n_real"`` and ``"n_imag"`` (complex
        refractive index; the imaginary part is the extinction coefficient),
        ``"loss"`` (energy loss function Im(-1/ε)) and ``"absorption"``
        (absorption coefficient in m⁻¹).
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
    """Write the real part of the refractive index to ``n_real.dat``.

    Parameters
    ----------
    data : dict of str to numpy.ndarray
        Calculated optical properties; must contain the key ``"n_real"``.
    energies : numpy.ndarray
        Incident photon energies in eV for the optical property data.
    directory : str or Path
        Output directory where the file is written.
    """
    out_path = Path(directory) / "n_real.dat" if directory else Path("n_real.dat")
    out = np.stack((energies, data["n_real"]), axis=1)
    np.savetxt(out_path, out, header="energy(eV) n_real")


def print_absorption_file(
        data: dict[str, NDArray], energies: NDArray, directory: str | Path
) -> None:
    """Write the absorption coefficient in cm⁻¹ to ``absorption.dat``.

    Parameters
    ----------
    data : dict of str to numpy.ndarray
        Calculated optical properties; must contain the key ``"absorption"``
        in m⁻¹.
    energies : numpy.ndarray
        Incident photon energies in eV for the optical property data.
    directory : str or Path
        Output directory where the file is written.
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


def generate_absorption(
        optics_directory: str | Path,
        out_directory: str | Path | None = None,
        code: str = "vasp",
        seedname: str | None = None,
) -> None:
    """Generate and write the absorption coefficient from an optics calculation.

    Parameters
    ----------
    optics_directory : str or Path
        Directory containing the dielectric data: ``vasprun.xml`` for VASP,
        an OptaDOS ``<seed>_epsilon.dat`` for CASTEP.
    out_directory : str or Path or None, optional
        Directory the ``absorption.dat`` file is written into. Default is
        None, which writes it beside the file it was derived from.
    code : str, optional
        Which code produced the data, ``"vasp"`` or ``"castep"``. Default is
        ``"vasp"``.
    seedname : str or None, optional
        CASTEP seed naming the epsilon file. Default is None, which globs
        for a single ``*_epsilon.dat``. Ignored for VASP.
    """
    if out_directory is None:
        out_directory = optics_directory

    filename = _resolve_optics_file(optics_directory, code, seedname)

    _, _, eps_full, _, energies = calc_dielectric(filename, code=code)
    data = calc_absorption(eps_full, energies)
    print_absorption_file(data, energies, out_directory)


def generate_n_real(
        optics_directory: str | Path,
        out_directory: str | Path | None = None,
        code: str = "vasp",
        seedname: str | None = None,
) -> None:
    """Generate and write the real refractive index from an optics calculation.

    Parameters
    ----------
    optics_directory : str or Path
        Directory containing the dielectric data: ``vasprun.xml`` for VASP,
        an OptaDOS ``<seed>_epsilon.dat`` for CASTEP.
    out_directory : str or Path or None, optional
        Directory the ``n_real.dat`` file is written into. Default is None,
        which writes it beside the file it was derived from.
    code : str, optional
        Which code produced the data, ``"vasp"`` or ``"castep"``. Default is
        ``"vasp"``.
    seedname : str or None, optional
        CASTEP seed naming the epsilon file. Default is None, which globs
        for a single ``*_epsilon.dat``. Ignored for VASP.
    """
    if out_directory is None:
        out_directory = optics_directory

    filename = _resolve_optics_file(optics_directory, code, seedname)

    _, _, eps_full, _, energies = calc_dielectric(filename, code=code)
    data = calc_absorption(eps_full, energies)
    print_n_real_file(data, energies, out_directory)


def plot_absorption(
        optics_directory: str | Path, xmax: float = 4, xmin: float = 0, save: bool = False,
        out_directory: str | Path = ".", code: str = "vasp", seedname: str | None = None
) -> None:
    """Plot the optical absorption spectrum from an optics calculation.

    Parameters
    ----------
    optics_directory : str or Path
        Directory containing the dielectric data: ``vasprun.xml`` for VASP,
        an OptaDOS ``<seed>_epsilon.dat`` for CASTEP.
    xmax : float, optional
        Maximum energy in eV shown on the x-axis. Default is ``4``.
    xmin : float, optional
        Minimum energy in eV shown on the x-axis. Default is ``0``.
    save : bool, optional
        Save the figure as ``absorption.png``. Default is ``False``.
    out_directory : str or Path, optional
        Directory the figure is written into when ``save`` is ``True``.
        Default is ``"."``, the current working directory.
    code : str, optional
        Which code produced the data, ``"vasp"`` or ``"castep"``. Default is
        ``"vasp"``.
    seedname : str or None, optional
        CASTEP seed naming the epsilon file. Default is None, which globs
        for a single ``*_epsilon.dat``. Ignored for VASP.
    """
    filename = _resolve_optics_file(optics_directory, code, seedname)
    eps_inf, eps_inf_tensor, eps_full, eps_imag, energies = calc_dielectric(filename, code=code)
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
        plt.savefig(Path(out_directory) / "absorption.png", dpi=700)

    plt.show()


def _spectrum_select(spectrum_type: str) -> tuple[NDArray, NDArray, bool]:
    """Select and load a solar or illuminant spectrum.

    Parameters
    ----------
    spectrum_type : str
        Identifier for the spectrum to load. ``"AM1.5"`` uses the standard
        AM1.5G spectrum bundled with the SLME package; anything else is
        loaded with ``load_spectrum``.

    Returns
    -------
    sol_wl : numpy.ndarray
        Wavelengths of the spectrum in nm.
    sol_irr : numpy.ndarray
        Spectral irradiance in W m⁻² nm⁻¹.
    use_slme : bool
        True if the built-in AM1.5G spectrum was used.
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
    """Convert a spectral irradiance distribution into photon flux.

    Parameters
    ----------
    sol_wl : numpy.ndarray
        Wavelengths of the spectrum in nm.
    sol_irr : numpy.ndarray
        Spectral irradiance in W m⁻² nm⁻¹.

    Returns
    -------
    sol_wl_m : numpy.ndarray
        Wavelengths converted to m.
    sol_phot_flux : numpy.ndarray
        Photon flux in photons m⁻² s⁻¹ nm⁻¹.
    """
    sol_wl_m = sol_wl * 1e-9  # Convert wavelength to meters
    sol_phot_flux = sol_irr * (sol_wl_m / (_h * _c))  # photons m-2 s-1 nm-1

    return sol_wl_m, sol_phot_flux


def _calc_incident_power(sol_irr: NDArray, sol_wl: NDArray) -> float:
    """Calculate the total incident power from a spectral irradiance distribution.

    Parameters
    ----------
    sol_irr : numpy.ndarray
        Spectral irradiance in W m⁻² nm⁻¹.
    sol_wl : numpy.ndarray
        Wavelengths for the irradiance spectrum in nm.

    Returns
    -------
    float
        Total incident power density in W m⁻², integrated over wavelength.
    """
    power_in = simpson(sol_irr, x=sol_wl)  # W m-2

    return power_in


def _bb_per_eV(E_eV: NDArray) -> NDArray:
    """Compute the blackbody photon flux spectrum in energy space.

    Evaluated at the module temperature ``_T``.

    Parameters
    ----------
    E_eV : numpy.ndarray
        Photon energies in eV.

    Returns
    -------
    numpy.ndarray
        Blackbody photon flux in photons m⁻² s⁻¹ eV⁻¹.
    """
    # blackbody photon flux in energy space [photons m-2 s-1 eV-1] for pe integral

    E_J = E_eV * _e
    exp = np.clip(E_eV / ((_k / _e) * _T), 0, 700)
    return (2 * E_J ** 2) / (_h ** 3 * _c ** 2) / (np.exp(exp) - 1.0 + 1e-300) * _e


def _bb_per_wl(sol_wl_m: NDArray) -> NDArray:
    """Compute the blackbody photon flux spectrum in wavelength space.

    Evaluated at the module temperature ``_T``.

    Parameters
    ----------
    sol_wl_m : numpy.ndarray
        Wavelengths in m.

    Returns
    -------
    numpy.ndarray
        Blackbody photon flux in photons m⁻² s⁻¹ m⁻¹.
    """
    # blackbody photon flux in wavelength space [photons m-2 s-1 m-1]
    bb_irr = (2 * _h * _c ** 2 / sol_wl_m ** 5) / (np.exp(_h * _c / (sol_wl_m * _k * _T)) - 1.0)
    bb_phot_wl = bb_irr * (sol_wl_m / (_h * _c))

    return bb_phot_wl


def _n_real_abs_fit(
        abs_file: str | Path, n_real_file: str | Path
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Load absorption data and interpolate the refractive index onto its energy grid.

    Parameters
    ----------
    abs_file : str or Path
        File containing absorption data: energy in eV against absorption
        coefficient in cm⁻¹.
    n_real_file : str or Path
        File containing real refractive index data: energy in eV against
        n_real.

    Returns
    -------
    energy_abs : numpy.ndarray
        Energy grid in eV from the absorption dataset.
    alpha_cm : numpy.ndarray
        Absorption coefficient in cm⁻¹.
    alpha_m : numpy.ndarray
        Absorption coefficient converted to m⁻¹.
    n_real : numpy.ndarray
        Real refractive index interpolated onto the absorption energy grid.
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
    """Interpolate the absorption coefficient onto a solar wavelength grid.

    Converts the absorption data from energy to wavelength space and enforces
    a cutoff: no absorption at wavelengths beyond the direct band gap.

    Parameters
    ----------
    energy_abs : numpy.ndarray
        Energy grid in eV for the absorption data.
    alpha_m : numpy.ndarray
        Absorption coefficient in m⁻¹.
    direct_gap : float
        Direct band gap energy in eV.
    sol_wl : numpy.ndarray
        Solar spectrum wavelength grid in nm.

    Returns
    -------
    numpy.ndarray
        Absorption coefficient on the solar wavelength grid, zero beyond the
        band-gap cutoff wavelength.
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
        out_directory: str | Path = ".",
) -> None:
    """Generate the efficiency-versus-thickness plot for the Blank and SLME models.

    Loads the absorption and refractive index data, selects a spectrum,
    computes photon fluxes, and evaluates the efficiency models across a
    range of thicknesses.

    Parameters
    ----------
    optics_directory : str or Path
        Directory containing the optical data files ``absorption.dat`` and
        ``n_real.dat``.
    direct_gap : float
        Direct band gap energy in eV.
    indirect_gap : float
        Indirect band gap energy in eV.
    spectrum_type : str, optional
        Type of solar spectrum to use. Default is ``"AM1.5"``.
    Qi : float, optional
        Internal quantum efficiency factor. Default is ``1.0``.
    n : float, optional
        Refractive index used in the model calculations. Default is ``3.5``.
    thickness_range : numpy.ndarray or None, optional
        Thickness values in m to evaluate. Default is None, which uses a
        logarithmic range.
    save : bool, optional
        Save the figure as ``slme.png``. Default is ``False``.
    out_directory : str or Path, optional
        Directory the figure is written into when ``save`` is ``True``.
        Default is ``"."``, the current working directory.
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

    plot_blank(use_slme, thickness_range, eff_slme, eff_lam, eff_flat, linestyle, save, out_directory)


def power_efficiency(
        A_E: NDArray, energy_abs: NDArray, n_real: NDArray, alpha_m: NDArray, d: float
) -> float:
    """Compute the power conversion efficiency using a spectral absorption model.

    Follows the detailed-balance framework of Blank et al., weighting the
    absorptance by the blackbody photon flux.

    Parameters
    ----------
    A_E : numpy.ndarray
        Absorptance evaluated on the energy grid.
    energy_abs : numpy.ndarray
        Energy grid in eV.
    n_real : numpy.ndarray
        Real refractive index on the same energy grid.
    alpha_m : numpy.ndarray
        Absorption coefficient in m⁻¹.
    d : float
        Material thickness in m.

    Returns
    -------
    float
        Power efficiency, dimensionless and capped at 1.0.
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
    """Calculate the power conversion efficiency at one thickness.

    Combines optical absorption, radiative recombination limits and external
    luminescence efficiency within a detailed-balance framework.

    Parameters
    ----------
    d : float
        Material thickness in m.
    A_sol : numpy.ndarray
        Absorptance on the solar wavelength grid.
    A_E : numpy.ndarray
        Absorptance on the energy grid.
    energy_abs : numpy.ndarray
        Energy grid in eV.
    n_real : numpy.ndarray
        Real refractive index on the energy grid.
    alpha_m : numpy.ndarray
        Absorption coefficient in m⁻¹.
    bb_phot_wl : numpy.ndarray
        Blackbody photon flux in wavelength space.
    sol_wl_m : numpy.ndarray
        Solar wavelength grid in m.
    sol_phot_flux : numpy.ndarray
        Solar photon flux in wavelength space.
    sol_wl : numpy.ndarray
        Solar wavelength grid in nm.
    Qi : float
        Internal quantum efficiency factor.
    power_in : float
        Incident solar power density in W m⁻².

    Returns
    -------
    float
        Power conversion efficiency in % at the optimal operating point.
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
        """Diode-law current density at voltage ``V``."""
        return Jsc - J0 * (np.exp(_e * V / (_k * _T)) - 1.0)

    def Pfn(V: float) -> float:
        """Output power density at voltage ``V``."""
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
    """Compute thickness-dependent power conversion efficiencies for several models.

    Evaluates efficiency against thickness with two absorption models
    (flat Beer-Lambert and Lambertian), and with SLME when requested.

    Parameters
    ----------
    thickness_range : numpy.ndarray or None
        Thickness values in m. If None, a logarithmic range from 1e-8 m to
        1e-3 m is used.
    alpha_m : numpy.ndarray
        Absorption coefficient in m⁻¹ on the energy grid.
    use_slme : bool
        Whether to also compute the SLME efficiency.
    n : float
        Refractive index used in the optical model.
    energy_abs : numpy.ndarray
        Energy grid in eV.
    alpha_cm : numpy.ndarray
        Absorption coefficient in cm⁻¹, used for SLME.
    direct_gap : float
        Direct band gap energy in eV.
    indirect_gap : float
        Indirect band gap energy in eV.
    n_real : numpy.ndarray
        Real refractive index on the energy grid.
    bb_phot_wl : numpy.ndarray
        Blackbody photon flux in wavelength space.
    sol_wl_m : numpy.ndarray
        Solar wavelength grid in m.
    sol_phot_flux : numpy.ndarray
        Solar photon flux in wavelength space.
    sol_wl : numpy.ndarray
        Solar wavelength grid in nm.
    Qi : float
        Internal quantum efficiency factor.
    power_in : float
        Incident solar power density in W m⁻².

    Returns
    -------
    eff_flat : list of float
        Efficiencies from the flat Beer-Lambert absorption model.
    eff_lam : list of float
        Efficiencies from the Lambertian (interference-enhanced) model.
    eff_slme : list of float
        SLME efficiencies; empty if ``use_slme`` is False.
    thickness_range : numpy.ndarray
        Thickness values used for the evaluation.
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
        out_directory: str | Path = ".",
) -> None:
    """Plot the thickness-dependent maximum efficiency for each optical model.

    Compares the SLME model (when enabled) with the Blank Lambertian and
    flat Beer-Lambert models.

    Parameters
    ----------
    use_slme : bool
        Whether SLME results are included and plotted.
    thickness_range : numpy.ndarray
        Film thickness values in m.
    eff_slme : list of float
        SLME efficiencies; may be empty when unused.
    eff_lam : list of float
        Efficiencies from the Lambertian optical model.
    eff_flat : list of float
        Efficiencies from the flat Beer-Lambert model.
    linestyle : str
        Matplotlib line style for the flat-model curve.
    save : bool
        Save the figure as ``slme.png``.
    out_directory : str or Path, optional
        Directory the figure is written into when ``save`` is ``True``.
        Default is ``"."``, the current working directory.
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
        plt.savefig(Path(out_directory) / "slme.png", dpi=700)
    plt.show()
