import logging

from matplotlib import pyplot as plt
from sumo.cli.dosplot import dosplot

logging.getLogger('matplotlib.font_manager').disabled = True
import numpy as np
from numpy.typing import NDArray

import warnings
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from pymatgen.io.vasp import Vasprun
from pymatgen.core.structure import Structure
from solphin.vasp_inputs import write_vasp_calculation
from pymatgen.io.vasp.inputs import Kpoints
import scipy.constants as sc
from scipy.constants import physical_constants as pc

HBAR = sc.hbar  # J·s
M_E = pc["atomic unit of mass"][0]  # kg
EV = sc.e  # J per eV
MIN_DOS_FIT_R2 = 0.80
MIN_DOS_FIT_POINTS = 10


def plot_dos(filename, xmin=-3, xmax=3, gaussian=0.05, save=False):
    """
    Plots the electronic density of states (DOS) from a calculation output file.

    This function generates a density of states plot using the provided DOS data
    file and visualizes it within a predefined plotting style. Optionally, the
    plot can also be saved as a PDF file.

    Parameters:
        filename(str): Path to the DOS data file to be plotted.

        xmin(float, optional): Minimum energy value shown on the x-axis in eV.
            Default is -3.

        xmax(float, optional): Maximum energy value shown on the x-axis in eV.
            Default is 3.

        gaussian(float, optional): Width of the Gaussian broadening applied to
            the DOS in eV. Default is 0.05.

        save(bool, optional): If True, saves the generated figure as
            "dos.png". Default is False.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(5, 3), dpi=150)
    dosplot(filename=filename, xmin=xmin, xmax=xmax, gaussian=gaussian, plt=plt)

    if save:
        plt.savefig("dos.png")

    plt.show()
    return


'''
Density of states effect mass classes.
'''


@dataclass
class _DOSEffectiveMassResult:
    carrier: str
    m_eff_rel: float
    m_eff_si: float
    fit_quality: float
    n_points: int
    E_edge: float
    energy_window: float
    fit_coefficient: float

    @property
    def E_c(self):
        return self.E_edge

    def __str__(self):
        edge_label = (
            "CBM"
            if self.carrier == "electrons"
            else "VBM"
        )

        sub = (
            "ₑ"
            if self.carrier == "electrons"
            else "ₕ"
        )

        lines = [
            "",
            "=" * 60,
            f" DOS Effective Mass ({self.carrier.capitalize()})",
            "=" * 60,
            f" Band edge ({edge_label}) : {self.E_edge:.6f} eV",
            f"  Energy window     : {self.energy_window:.4f} eV",
            f"  Points used       : {self.n_points}",
            f"  Fit quality       : {self.fit_quality:.6f} R²",
            "",
            f"  DOS effective mass: {self.m_eff_rel:.6f} m{sub}",
            f"                    : {self.m_eff_si:.6e} kg",
            "=" * 60,
            "",
        ]

        return "\n".join(lines)


@dataclass
class DOSResult:
    fit_quality_e: Optional[float]
    fit_quality_h: Optional[float]
    cell_volume_m3: float
    carrier: str
    final_result: float
    em_electrons: Optional[_DOSEffectiveMassResult] = None
    em_holes: Optional[_DOSEffectiveMassResult] = None
    cbm: Optional[float] = None
    vbm: Optional[float] = None

    @property
    def em_result(self) -> Optional[_DOSEffectiveMassResult]:
        if self.carrier == "electrons":
            return self.em_electrons

        return self.em_holes

    def __str__(self):
        return _format_dos_summary(self)


def _load_dos(
        dos_vasprun: str
):
    """
    Loads the electronic density of states (DOS) from a VASP calculation output file.

    This function parses a VASP ``vasprun.xml`` file, extracts the complete
    density of states, and calculates the total DOS by summing the density
    contributions from all available spin channels.

    Parameters:
        dos_vasprun(str): Path to the VASP ``vasprun.xml`` file containing the
            density of states data.

    Returns:
        tuple: A tuple containing the parsed VASP data and density of states
            information:

            vr(Vasprun): Parsed VASP calculation output.

            cdos(CompleteDos): Complete density of states object containing the
                total and projected DOS information.

            energies(np.ndarray): Energy values corresponding to the DOS in eV.

            densities(np.ndarray): Total electronic density of states obtained
                by summing over all spin channels.
    """

    vr = Vasprun(
        dos_vasprun,
        parse_dos=True,
        parse_eigen=False
    )

    cdos = vr.complete_dos

    energies = np.asarray(
        cdos.energies,
        dtype=float
    )

    densities = np.zeros_like(
        energies,
        dtype=float
    )

    for spin_density in cdos.densities.values():
        densities += np.asarray(
            spin_density,
            dtype=float
        )

    return vr, cdos, energies, densities


def _get_band_edge(cdos, carrier, energies, energy_window, densities):
    """
    Extracts the electronic density of states near a selected band edge.

    This function determines the conduction band minimum (CBM) or valence band
    maximum (VBM) depending on the specified carrier type and extracts the DOS
    within a given energy window from the corresponding band edge. Energies are
    returned relative to the selected band edge.

    Parameters:
        cdos(CompleteDos): Complete density of states object containing the
            conduction and valence band edge information.

        carrier(str): Type of charge carrier. If "electrons", the conduction
            band minimum is used. Otherwise, the valence band maximum is used.

        energies(np.ndarray): Energy values corresponding to the DOS in eV.

        energy_window(float): Energy range from the selected band edge over
            which the DOS is extracted in eV.

        densities(np.ndarray): Electronic density of states corresponding to
            the supplied energy values.

    Returns:
        tuple: A tuple containing the band-edge energy and DOS data within the
            selected energy window:

            E_edge(float): Energy of the selected band edge in eV.

            delta_E_ev(np.ndarray): Energy values relative to the selected band
                edge in eV.

            dos(np.ndarray): Electronic density of states within the selected
                energy window.
    """

    cbm, vbm = cdos.get_cbm_vbm()

    if carrier == "electrons":

        E_edge = cbm

        mask = (
                (energies > E_edge)
                & (energies <= E_edge + energy_window)
        )

        delta_E_ev = (
                energies[mask] - E_edge
        )

    else:

        E_edge = vbm

        mask = (
                (energies < E_edge)
                & (energies >= E_edge - energy_window)
        )

        delta_E_ev = (
                E_edge - energies[mask]
        )

    dos = densities[mask]

    return E_edge, delta_E_ev, dos


def _clean_dos_values(delta_E_ev, dos, min_dos, energy_window):
    """
    Cleans and validates density of states (DOS) data near a band edge.

    This function removes invalid DOS data points, including non-finite values,
    non-positive relative energies, and DOS values below a specified threshold.
    It also ensures that enough valid data points remain for subsequent
    analysis.

    Parameters:
        delta_E_ev(np.ndarray): Energy values relative to the selected band edge
            in eV.

        dos(np.ndarray): Electronic density of states corresponding to the
            relative energy values.

        min_dos(float): Minimum DOS value required for a data point to be
            retained.

        energy_window(float): Energy range from the band edge used to select the
            DOS data in eV. Used when reporting insufficient valid data points.

    Returns:
        tuple: A tuple containing the cleaned energy and DOS arrays:

            delta_E_ev(np.ndarray): Valid relative energy values in eV.

            dos(np.ndarray): Valid electronic density of states values.

    Raises:
        ValueError: If fewer than three usable DOS data points remain after
            filtering.
    """

    good = (
            np.isfinite(delta_E_ev)
            & np.isfinite(dos)
            & (delta_E_ev > 0)
            & (dos > min_dos)
    )

    delta_E_ev = delta_E_ev[good]
    dos = dos[good]

    if len(dos) < 3:
        raise ValueError(
            f"Only {len(dos)} usable DOS points lie within "
            f"{energy_window:.3f} eV of the band edge."
        )

    return delta_E_ev, dos


def _convert_dos(vr, delta_E_ev, dos):
    """
    Converts density of states (DOS) data from VASP units to SI units.

    This function converts relative energy values from electronvolts to joules
    and normalizes the DOS by the final unit-cell volume. The resulting DOS is
    expressed per joule per cubic metre.

    Parameters:
        vr(Vasprun): Parsed VASP calculation output containing the final
            structure and unit-cell volume.

        delta_E_ev(np.ndarray): Energy values relative to the selected band edge
            in eV.

        dos(np.ndarray): Electronic density of states in states per eV.

    Returns:
        tuple: A tuple containing the converted energy and DOS arrays:

            delta_E_J(np.ndarray): Energy values relative to the selected band
                edge in J.

            dos_si(np.ndarray): Electronic density of states in states per J
                per m^3.
    """

    volume_m3 = (
            vr.final_structure.volume
            * 1.0e-30
    )

    delta_E_J = (
            delta_E_ev * EV
    )

    dos_si = (
            dos
            / EV
            / volume_m3
    )

    return delta_E_J, dos_si


def _check_fit(A, x, y):
    """
    Evaluates the quality of a linear fit constrained to pass through the origin.

    This function calculates the predicted values from a linear model of the
    form y = A*x and evaluates the goodness of fit using the coefficient of
    determination (R^2).

    Parameters:
        A(float): Fitted proportionality constant defining the slope of the
            linear model.

        x(np.ndarray): Independent variable values used in the fit.

        y(np.ndarray): Observed dependent variable values to be compared with
            the fitted model.

    Returns:
        float: Coefficient of determination (R^2) describing the quality of the
            linear fit. A value of 1 indicates a perfect fit.
    """

    y_pred = (
            A * x
    )

    ss_res = np.sum(
        (y - y_pred) ** 2
    )

    ss_tot = np.sum(
        (y - np.mean(y)) ** 2
    )

    if ss_tot > 0:
        r2 = 1.0 - ss_res / ss_tot
    else:
        r2 = 1.0

    return r2


def _calculate_DOS(delta_E_J, dos_si):
    """
    Calculates the density-of-states effective mass from DOS data near a band edge.

    This function fits the density of states to the square-root energy
    dependence expected for a three-dimensional parabolic band and uses the
    fitted coefficient to calculate the DOS effective mass. The quality of the
    fit is evaluated using the coefficient of determination (R^2).

    Parameters:
        delta_E_J(np.ndarray): Energy values relative to the selected band edge
            in J.

        dos_si(np.ndarray): Electronic density of states in states per J per
            m^3.

    Returns:
        tuple: A tuple containing the calculated effective mass and fitting
            information:

            m_eff_rel(float): DOS effective mass expressed relative to the free
                electron mass.

            m_eff_si(float): DOS effective mass in kg.

            r2(float): Coefficient of determination (R^2) describing the
                quality of the DOS fit.

            y(np.ndarray): DOS values used as the dependent variable in the
                fit.

            A(float): Fitted coefficient relating the DOS to the square root of
                energy.

    Raises:
        ValueError: If the energy data have zero spread or if the fitted DOS
            coefficient is non-positive.
    """

    x = np.sqrt(
        delta_E_J
    )

    y = dos_si

    denominator = np.dot(
        x,
        x,
    )

    if denominator == 0:
        raise ValueError(
            "Cannot fit DOS effective mass: zero energy spread."
        )

    A = (
            np.dot(x, y)
            / denominator
    )

    if A <= 0:
        raise ValueError(
            "Fitted DOS coefficient is non-positive."
        )

    m_eff_si = (
            HBAR ** 2
            / 2.0
            * (
                    2.0
                    * np.pi ** 2
                    * A
            ) ** (2.0 / 3.0)
    )

    m_eff_rel = (
            m_eff_si / M_E
    )

    r2 = _check_fit(A, x, y)

    return m_eff_rel, m_eff_si, r2, y, A


def get_dos_effective_mass(
        dos_vasprun: str,
        carrier: str = "electrons",
        energy_window: float = 0.15,
        min_dos: float = 0.0,
) -> _DOSEffectiveMassResult:
    """
    Calculates the density-of-states effective mass for electrons or holes.

    This function extracts the electronic density of states near the selected
    band edge from a VASP ``vasprun.xml`` file, filters and converts the DOS
    data to SI units, and fits the expected square-root energy dependence for a
    three-dimensional parabolic band. The resulting fit is used to calculate
    the DOS effective mass and assess the quality of the fit.

    Parameters:
        dos_vasprun(str): Path to the VASP ``vasprun.xml`` file containing the
            density of states data.

        carrier(str, optional): Type of charge carrier for which the effective
            mass is calculated. Must be either "electrons" or "holes". Default
            is "electrons".

        energy_window(float, optional): Energy range from the selected band edge
            over which the DOS is used for fitting in eV. Default is 0.15.

        min_dos(float, optional): Minimum DOS value required for a data point to
            be included in the fit. Default is 0.0.

    Returns:
        _DOSEffectiveMassResult: Object containing the calculated DOS effective
            mass and fitting information, including the effective mass relative
            to the free electron mass, the effective mass in kg, fit quality,
            number of fitted points, selected band-edge energy, fitting energy
            window, and fitted DOS coefficient.

    Raises:
        ValueError: If ``carrier`` is not "electrons" or "holes", if fewer than
            three usable DOS points remain after filtering, if the selected
            energy data have zero spread, or if the fitted DOS coefficient is
            non-positive.
    """

    if carrier not in ("electrons", "holes"):
        raise ValueError(
            f"Carrier must be 'electrons' or 'holes' recieved: {carrier!r}"
        )

    vr, cdos, energies, densities = _load_dos(dos_vasprun)

    E_edge, delta_E_ev, dos = _get_band_edge(cdos, carrier, energies, energy_window, densities)

    delta_E_ev, dos = _clean_dos_values(delta_E_ev, dos, min_dos, energy_window)

    delta_E_J, dos_si = _convert_dos(vr, delta_E_ev, dos)

    m_eff_rel, m_eff_si, r2, y, A = _calculate_DOS(delta_E_J, dos_si)

    return _DOSEffectiveMassResult(
        carrier=carrier,

        m_eff_rel=float(m_eff_rel),
        m_eff_si=float(m_eff_si),

        fit_quality=float(r2),
        n_points=len(y),

        E_edge=float(E_edge),
        energy_window=float(energy_window),

        fit_coefficient=float(A),
    )


def test_dos_mass_windows(
        dos_vasprun: str,
        carrier: str = "electrons",
        windows=(0.05, 0.10, 0.15, 0.20, 0.30),
        min_dos: float = 0.0,
):
    """
    Tests the sensitivity of the DOS effective mass to the fitting energy window.

    This function calculates the density-of-states effective mass over a range
    of fitting windows and prints a convergence table containing the fitted
    effective mass, fit quality, and number of DOS points used for each window.
    Failed fits are reported without interrupting the remaining calculations.

    Parameters:
        dos_vasprun(str): Path to the VASP ``vasprun.xml`` file containing the
            density of states data.

        carrier(str, optional): Type of charge carrier for which the effective
            mass is calculated. Must be either "electrons" or "holes". Default
            is "electrons".

        windows(tuple, optional): Sequence of energy windows in eV used to test
            the sensitivity of the DOS effective-mass fit. Default is
            (0.05, 0.10, 0.15, 0.20, 0.30).

        min_dos(float, optional): Minimum DOS value required for a data point to
            be included in each fit. Default is 0.0.

    Returns:
        list: A list of ``_DOSEffectiveMassResult`` objects corresponding to
            each successful fitting window. Windows for which the fit fails are
            not included.

    Notes:
        A convergence table is printed to standard output showing the fitting
        window, DOS effective mass relative to the free electron mass, R^2
        value, and number of fitted DOS points for each successful calculation.
    """

    print("")
    print("=" * 78)

    print(
        f"  FOM DOS effective-mass convergence "
        f"({carrier})"
    )

    print("=" * 78)

    print(
        f"{'Window / eV':>12} "
        f"{'m_DOS / m_e':>15} "
        f"{'R²':>12} "
        f"{'N':>8}"
    )

    print("-" * 78)

    results = []

    for window in windows:

        try:

            result = get_dos_effective_mass(
                dos_vasprun=dos_vasprun,
                carrier=carrier,
                energy_window=window,
                min_dos=min_dos,
            )

            print(
                f"{window:12.4f} "
                f"{result.m_eff_rel:15.6f} "
                f"{result.fit_quality:12.6f} "
                f"{result.n_points:8d}"
            )

            results.append(
                result
            )

        except ValueError as exc:

            print(
                f"{window:12.4f} "
                f"FAILED: {exc}"
            )

    print("=" * 78)
    print("")

    return results


def _format_em_table(
        em: _DOSEffectiveMassResult,
        edge: float,
        is_dos_carrier: bool,
        fit: Optional[float],
) -> list:
    """
    Formats DOS effective-mass results for display in a summary table.

    This function converts a DOS effective-mass result into a list of formatted
    text lines containing the carrier type, fitted band edge, effective mass,
    fit quality, number of fitted points, and fitting energy window. The
    selected DOS carrier can optionally be marked as the Crovetto DOS mass.

    Parameters:
        em(_DOSEffectiveMassResult): DOS effective-mass result containing the
            calculated mass and fitting information.

        edge(float): Energy of the band edge at which the effective mass was
            fitted in eV.

        is_dos_carrier(bool): If True, marks the effective mass as the Crovetto
            DOS mass in the formatted output.

        fit(float, optional): Fit quality expressed as the coefficient of
            determination (R²). If None, the fit quality is displayed as
            "N/A".

    Returns:
        list: A list of formatted strings containing the DOS effective-mass
            results for display.
    """

    edge_label = (
        "CBM"
        if em.carrier == "electrons"
        else "VBM"
    )

    sub = (
        "ₑ"
        if em.carrier == "electrons"
        else "ₕ"
    )

    marker = (
        "  ← FOM DOS mass"
        if is_dos_carrier
        else ""
    )

    fit_text = (
        f"{fit:.6f}"
        if fit is not None
        else "N/A"
    )

    return [
        f"  {em.carrier.capitalize()} "
        f"(fitted at {edge_label}: {edge:.3f} eV)",

        f"  {'DOS effective mass':<22}: "
        f"{em.m_eff_rel:.6f} m{sub}"
        f"  ({em.m_eff_si:.3e} kg)"
        f"{marker}",

        f"  {'Fit quality':<22}: "
        f"{fit_text} R²",

        f"  {'Points fitted':<22}: "
        f"{em.n_points}",

        f"  {'Energy window':<22}: "
        f"{em.energy_window:.4f} eV",

        "",
    ]


def _format_dos_summary(
        result: "DOSResult",
) -> str:
    """
    Formats a complete density-of-states (DOS) result summary for display.

    This function generates a formatted text summary containing the primary
    carrier type, corresponding band-edge energy, cell volume, and Crovetto DOS
    effective-mass results for electrons and holes. If an effective mass was
    not calculated for a carrier type, this is indicated in the output.

    Parameters:
        result(DOSResult): DOS result containing the carrier type, band-edge
            energies, cell volume, electron and hole effective masses, and
            associated fit-quality information.

    Returns:
        str: Formatted multi-line string summarizing the DOS calculation and
            available effective-mass results.
    """

    edge_label = (
        "CBM"
        if result.carrier == "electrons"
        else "VBM"
    )

    edge_value = (
        result.cbm
        if result.carrier == "electrons"
        else result.vbm
    )

    lines = [
        "",
        "=" * 60,
        "  DOS Result Summary",
        "=" * 60,
        f"  Primary carrier     : "
        f"{result.carrier.capitalize()}",
        f"  Band edge ({edge_label})     : "
        f"{edge_value:.3f} eV",
        f"  Cell volume         : "
        f"{result.cell_volume_m3:.3e} m³",
        "",
        "  ── FOM DOS Effective Masses "
        + "─" * 23,
    ]

    if result.em_electrons is not None:

        lines += _format_em_table(
            result.em_electrons,
            result.cbm,
            is_dos_carrier=(
                    result.carrier == "electrons"
            ),
            fit=result.fit_quality_e,
        )

    else:

        lines.append(
            "  Electrons: DOS effective mass not calculated."
        )

    lines.append("")

    if result.em_holes is not None:

        lines += _format_em_table(
            result.em_holes,
            result.vbm,
            is_dos_carrier=(
                    result.carrier == "holes"
            ),
            fit=result.fit_quality_h,
        )

    else:

        lines.append(
            "  Holes: DOS effective mass not calculated."
        )

    lines += [
        "=" * 60,
        "",
    ]

    return "\n".join(
        lines
    )


def print_dos_summary(
        result: "DOSResult",
) -> None:
    """
    Prints a formatted density-of-states (DOS) result summary.

    This function generates the complete DOS summary using the associated
    formatting function and prints it to standard output. The summary includes
    the primary carrier, band-edge information, cell volume, and available
    electron and hole DOS effective-mass results.

    Parameters:
        result(DOSResult): DOS result containing the calculated density-of-states
            and effective-mass information to be displayed.

    Returns:
        None
    """

    print(
        _format_dos_summary(
            result
        )
    )


def _check_dos_fit_quality(
        result: _DOSEffectiveMassResult,
        dos_vasprun: str,
        windows=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40),
        min_dos: float = 0.0,
) -> bool:
    """
    Checks whether a DOS effective-mass fit is sufficiently well resolved.

    This function evaluates the quality of a calculated DOS effective mass using
    predefined thresholds for the coefficient of determination and number of
    fitted DOS points. If the fit is poorly resolved, a warning is printed
    together with recommendations and a fitting-window convergence test.

    Parameters:
        result(_DOSEffectiveMassResult): DOS effective-mass result containing the
            fitted mass, fit quality, number of fitted points, and carrier type.

        dos_vasprun(str): Path to the VASP ``vasprun.xml`` file containing the
            density of states data. Used to test the fitting-window dependence
            when the fit is poorly resolved.

        windows(tuple, optional): Sequence of energy windows in eV used to test
            the sensitivity of a poorly resolved DOS effective-mass fit. Default
            is (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40).

        min_dos(float, optional): Minimum DOS value required for a data point to
            be included in the fitting-window tests. Default is 0.0.

    Returns:
        bool: True if the DOS effective-mass fit passes the quality check.
            False if the fit is poorly resolved and a warning is issued.
    """

    poor_fit = (
            result.fit_quality < MIN_DOS_FIT_R2
            and result.n_points < MIN_DOS_FIT_POINTS
    )

    if not poor_fit:
        return True

    print("")
    print("!" * 70)
    print("  WARNING: DOS EFFECTIVE-MASS FIT IS POORLY RESOLVED")
    print("!" * 70)

    print(
        f"  Carrier            : {result.carrier}"
    )

    print(
        f"  Fitted mass        : "
        f"{result.m_eff_rel:.6f} m_e"
    )

    print(
        f"  Fit quality        : "
        f"R² = {result.fit_quality:.6f}"
    )

    print(
        f"  DOS points fitted  : "
        f"{result.n_points}"
    )

    print("")
    print(
        "  This DOS calculation does not contain enough "
        "well-resolved"
    )
    print(
        "  near-edge DOS data to determine the Crovetto "
        "DOS effective"
    )
    print(
        "  mass reliably."
    )

    print("")
    print(
        "  Consider either:"
    )
    print(
        "    1. Re-running the DOS calculation with a "
        "denser regular k-point grid."
    )
    print(
        "    2. Supplying an alternative effective-mass "
        "estimate if a denser"
    )
    print(
        "       hybrid-functional calculation is "
        "prohibitively expensive."
    )

    print("")
    print(
        "  The current fitted mass should therefore be "
        "treated as provisional."
    )

    print("")
    print(
        "  Fitting-window behaviour:"
    )

    test_dos_mass_windows(
        dos_vasprun=dos_vasprun,
        carrier=result.carrier,
        windows=windows,
        min_dos=min_dos,
    )

    return False


def compute_dos(
        dos_vasprun: str,
        m_eff: Optional[float] = None,
        carrier: str = "electrons",
        energy_window: float = 0.15,
        min_dos: float = 0.0,
) -> DOSResult:
    """
    Computes density-of-states information and the corresponding effective mass.

    This function parses a VASP ``vasprun.xml`` file, extracts the conduction
    and valence band edges, and determines the DOS effective masses for
    electrons and holes. If an effective mass is supplied explicitly, that
    value is used as the final result instead of calculating a Crovetto DOS
    effective mass. Calculated fits are also checked for sufficient quality.

    Parameters:
        dos_vasprun(str): Path to the VASP ``vasprun.xml`` file containing the
            density of states data.

        m_eff(float, optional): User-supplied effective mass expressed relative
            to the free electron mass. If provided, this value is used as the
            final effective mass and DOS effective-mass fitting is skipped.
            Default is None.

        carrier(str, optional): Primary charge carrier for which the final
            effective mass is selected. Must be either "electrons" or "holes".
            Default is "electrons".

        energy_window(float, optional): Energy range from each band edge over
            which the DOS effective-mass fits are performed in eV. Default is
            0.15.

        min_dos(float, optional): Minimum DOS value required for a data point to
            be included in the effective-mass fits. Default is 0.0.

    Returns:
        DOSResult: Object containing the cell volume, zero-referenced band-edge
            energies, selected effective mass, electron and hole DOS
            effective-mass results, and associated fit-quality information.

    Raises:
        ValueError: If ``carrier`` is not "electrons" or "holes", or if the
            requested carrier effective mass cannot be calculated when no
            user-supplied effective mass is provided.

    Warns:
        UserWarning: If the electron or hole DOS effective-mass calculation
            fails. The calculation for the other carrier is still attempted.
    """

    if carrier not in (
            "electrons",
            "holes",
    ):
        raise ValueError(
            "carrier must be "
            "'electrons' or 'holes'."
        )

    vr = Vasprun(
        dos_vasprun,
        parse_dos=True,
        parse_eigen=False,
    )

    cdos = (
        vr.complete_dos
    )

    vol_m3 = (
            vr.final_structure.volume
            * 1.0e-30
    )

    cbm, vbm = (
        cdos.get_cbm_vbm()
    )

    # Reference energies to VBM = 0
    cbm_zeroed = (
            cbm - vbm
    )

    vbm_zeroed = 0.0

    em_electrons = None
    em_holes = None

    fit_quality_e = None
    fit_quality_h = None

    if m_eff is None:

        print(
            "  Computing electron "
            "FOM DOS effective mass..."
        )

        try:

            em_electrons = (
                get_dos_effective_mass(
                    dos_vasprun=dos_vasprun,
                    carrier="electrons",
                    energy_window=energy_window,
                    min_dos=min_dos,
                )
            )

            fit_quality_e = (
                em_electrons.fit_quality
            )

        except Exception as exc:

            warnings.warn(
                "Electron DOS effective-mass fit failed: "
                f"{exc}",
                UserWarning,
                stacklevel=2,
            )

        print(
            "  Computing hole "
            "FOM DOS effective mass..."
        )

        try:

            em_holes = (
                get_dos_effective_mass(
                    dos_vasprun=dos_vasprun,
                    carrier="holes",
                    energy_window=energy_window,
                    min_dos=min_dos,
                )
            )

            fit_quality_h = (
                em_holes.fit_quality
            )

        except Exception as exc:

            warnings.warn(
                "Hole DOS effective-mass fit failed: "
                f"{exc}",
                UserWarning,
                stacklevel=2,
            )

    if m_eff is not None:

        final_result = float(
            m_eff
        )

    elif carrier == "electrons":

        if em_electrons is None:
            raise ValueError(
                "Electron DOS effective mass could not "
                "be calculated."
            )

        final_result = (
            em_electrons.m_eff_rel
        )

    else:

        if em_holes is None:
            raise ValueError(
                "Hole DOS effective mass could not "
                "be calculated."
            )

        final_result = (
            em_holes.m_eff_rel
        )

    result = DOSResult(
        fit_quality_e=fit_quality_e,
        fit_quality_h=fit_quality_h,

        cell_volume_m3=vol_m3,

        carrier=carrier,

        final_result=final_result,

        em_electrons=em_electrons,
        em_holes=em_holes,

        cbm=cbm_zeroed,
        vbm=vbm_zeroed,
    )

    if m_eff is None:

        if carrier == "electrons":
            selected_em = em_electrons
        else:
            selected_em = em_holes

        if selected_em is not None:
            _check_dos_fit_quality(
                result=selected_em,
                dos_vasprun=dos_vasprun,
                min_dos=min_dos,
            )

    return result


'''
Density of states file generation
'''


def _generate_local_kpoints(
        k0_frac: NDArray,
        mesh: tuple,
        delta: float,
) -> Kpoints:
    """
    Generates a local reciprocal-space k-point mesh around a band-edge k-point.

    This function constructs a uniform three-dimensional grid of k-points
    centred on a specified fractional reciprocal-space coordinate. The grid
    extends by a specified displacement in each reciprocal direction and is
    returned as a VASP KPOINTS object in reciprocal-coordinate mode.

    Parameters:
        k0_frac(NDArray): Fractional reciprocal-space coordinates of the
            central k-point around which the local mesh is generated.

        mesh(tuple): Number of k-points along each reciprocal-space direction,
            specified as (nx, ny, nz).

        delta(float): Maximum fractional reciprocal-space displacement from the
            central k-point along each direction.

    Returns:
        Kpoints: VASP KPOINTS object containing the generated local reciprocal-
            space mesh with equal weights assigned to all k-points.
    """

    k0_frac = np.asarray(k0_frac, dtype=float)

    nx, ny, nz = mesh

    xs = np.linspace(-delta, delta, nx)
    ys = np.linspace(-delta, delta, ny)
    zs = np.linspace(-delta, delta, nz)

    pts = []

    for dx in xs:
        for dy in ys:
            for dz in zs:
                pts.append(
                    k0_frac + np.array([dx, dy, dz])
                )

    pts = np.asarray(pts)

    return Kpoints(
        comment="Local k-mesh around band edge",
        style=Kpoints.supported_modes.Reciprocal,
        num_kpts=len(pts),
        kpts=pts.tolist(),
        kpts_weights=[1.0] * len(pts),
    )


def write_local_kpoints(
        folder: str,
        k0_frac: NDArray,
        mesh: tuple,
        delta: float
):
    """
    Generates and writes a dense local VASP KPOINTS file around a band-edge k-point.

    This function generates a three-dimensional reciprocal-space k-point mesh
    centred on a specified band-edge k-point and writes the resulting mesh to a
    ``KPOINTS`` file in the specified calculation folder.

    Parameters:
        folder(str): Path to the calculation folder in which the generated
            ``KPOINTS`` file is written.

        k0_frac(NDArray): Fractional reciprocal-space coordinates of the
            central k-point, typically corresponding to the CBM, VBM, or a
            relevant direct band-gap location.

        mesh(tuple): Number of k-points along each reciprocal-space direction,
            specified as (nx, ny, nz).

        delta(float): Maximum fractional reciprocal-space displacement from the
            central k-point along each direction.

    Returns:
        None
    """

    kpoints = _generate_local_kpoints(k0_frac, mesh, delta)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    kp = Kpoints(
        comment="Local k-mesh for effective mass",
        style=Kpoints.supported_modes.Reciprocal,
        num_kpts=len(kpoints),
        kpts=kpoints.tolist(),
        kpts_weights=[1] * len(kpoints), )

    kp.write_file(f"{folder}/KPOINTS")


def write_eff_mass(
        k0_frac: NDArray,
        structure: Structure,
        functional: str,
        encut: int,
        folder: str = "eff_mass",
        mesh: tuple = (5, 5, 5),
        delta: float = 0.01,
):
    """
    Writes a VASP calculation setup for an effective-mass calculation.

    This function generates a dense local reciprocal-space k-point mesh around
    a specified band-edge k-point and uses it to prepare a VASP effective-mass
    calculation. The calculation is written using the requested functional,
    plane-wave cutoff energy, and effective-mass INCAR settings.

    Parameters:
        k0_frac(NDArray): Fractional reciprocal-space coordinates of the
            band-edge k-point around which the local k-point mesh is generated.

        structure(Structure): Crystal structure used to generate the VASP
            calculation input files.

        functional(str): VASP calculation recipe or exchange-correlation
            functional used to prepare the effective-mass calculation.

        encut(int): Plane-wave cutoff energy used for the VASP calculation in
            eV.

        folder(str, optional): Path to the folder in which the VASP calculation
            input files are written. Default is "eff_mass".

        mesh(tuple, optional): Number of k-points along each reciprocal-space
            direction, specified as (nx, ny, nz). Default is (5, 5, 5).

        delta(float, optional): Maximum fractional reciprocal-space displacement
            from the central k-point along each direction. Default is 0.01.

    Returns:
        None
    """

    kp = _generate_local_kpoints(
        k0_frac=k0_frac,
        mesh=mesh,
        delta=delta,
    )

    write_vasp_calculation(
        structure=structure,
        recipe=functional,
        out_dir=folder,
        patches=["eff_mass"],
        user_incar_settings={"ENCUT": encut, "ISYM": 0, "ICHARG": 0},
        user_kpoints_settings=kp,
    )
