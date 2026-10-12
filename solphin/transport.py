"""Carrier mobility from AMSET transport output, prepared for the Γₚᵥ figure of merit.

Γₚᵥ (Crovetto 2024) takes a single carrier mobility ``mu`` alongside the
doping density ``dop_density``, and the two are not independent: ionized
impurity scattering lowers the mobility as the doping rises, and free-carrier
screening of polar optical scattering can raise it. AMSET (Ganose *et al.*,
Nat. Commun. 12, 2222, 2021) computes μ(n, T) from first principles, so its
transport file can stand in for the hand-picked constant ``mu``.

Definitions that line up, and one that does not:

* AMSET's ``doping`` is the net free-carrier concentration |n - p| the Fermi
  level is solved for, negative for n-type and positive for p-type. That is
  Crovetto's ``dop_density`` (equation 3, n = |N_D - N_A|), so it is used as is,
  sign stripped. It is not a count of charged defects: those enter AMSET only
  through the ionized-impurity scattering, as compensation_factor x |n - p| /
  defect_charge, and neither setting is stored in the transport file.
* AMSET's mobility is the conductivity (drift) mobility of the *majority*
  carrier at each doping - electrons for the n-type entries, holes for the
  p-type ones - which is the μ = qτₛ/M of Crovetto's equation 5.
* Γₚᵥ assumes μₙ = μₚ and isotropic transport. AMSET gives one tensor per
  carrier type, so a choice has to be made; :func:`select_mobility` makes it
  explicit and reports it, defaulting to the lower ("limiting") of the two
  carriers' isotropic averages. This follows equation S2 of the Crovetto 2024
  supplementary material: the minority-carrier mobility under low injection,
  min(μₙ, μₚ) otherwise.

Only the transport file is read, as plain JSON or text, so AMSET itself does
not need to be installed.
"""

import gzip
import json
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import overload

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import ArrayLike, NDArray

from solphin.pv_fom import SAMPLED_RANGES

__all__ = [
    "CARRIER_CHOICES",
    "AmsetTransport",
    "MobilityModel",
    "load_amset_transport",
    "plot_mobility",
    "select_mobility",
]

#: Carrier choices accepted by :func:`select_mobility`.
CARRIER_CHOICES = ("limiting", "electrons", "holes")

# The name AMSET gives ionized impurity scattering in its mobility breakdown.
_IMP_LABEL = "IMP"

# Below this share of the total scattering, ionized impurities are taken as
# negligible, so the mobility is phonon-limited and flat in doping: holding the
# lowest computed value constant below the AMSET grid is then an accurate
# extrapolation rather than a guess.
_FLAT_EXTRAPOLATION_IMP_SHARE = 0.01

# Above this share, the compensation_factor and defect_charge AMSET was run with
# move the mobility noticeably, and neither is recorded in the transport file.
_SIGNIFICANT_IMP_SHARE = 0.10

# Largest-to-smallest eigenvalue ratio of the mobility tensor above which the
# isotropic average Γₚᵥ needs hides a real direction dependence.
_ANISOTROPY_RATIO = 1.5

_SINGULAR = {"electrons": "electron", "holes": "hole"}

# Relative tolerance when matching a requested temperature to the file's grid.
_TEMPERATURE_RTOL = 1e-6

# Text-format headers carry these six upper-triangle components, in this order.
_TRIU_COMPONENTS = ("xx", "xy", "xz", "yy", "yz", "zz")

_DEFAULT_DOP_RANGE: tuple[float, float] = SAMPLED_RANGES["dop_density"][:2]


@dataclass(frozen=True)
class AmsetTransport:
    """Transport data read from an AMSET ``transport_*.json`` or ``.txt`` file.

    Attributes
    ----------
    filename : Path
        The file the data was read from.
    doping : numpy.ndarray
        Net carrier concentrations in cm⁻³, shape (n_doping,), in AMSET's sign
        convention: negative for n-type (electrons), positive for p-type
        (holes).
    temperatures : numpy.ndarray
        Temperatures in K, shape (n_temperature,).
    fermi_levels : numpy.ndarray
        Fermi level in eV for each doping and temperature, shape
        (n_doping, n_temperature).
    mobility : dict of str to numpy.ndarray
        Mobility tensors in cm² V⁻¹ s⁻¹, each of shape
        (n_doping, n_temperature, 3, 3). ``"overall"`` is the total; any other
        keys (``"ADP"``, ``"IMP"``, ``"POP"``, ``"PIE"``, ...) are the
        mobilities limited by that scattering mechanism alone, present when
        AMSET ran with ``separate_mobility: true``.
    """

    filename: Path
    doping: NDArray
    temperatures: NDArray
    fermi_levels: NDArray
    mobility: dict[str, NDArray] = field(repr=False)

    @property
    def carriers(self) -> tuple[str, ...]:
        """Carrier types the file holds mobilities for, electrons first."""
        present = []

        if np.any(self.doping < 0):
            present.append("electrons")

        if np.any(self.doping > 0):
            present.append("holes")

        return tuple(present)

    @property
    def mechanisms(self) -> tuple[str, ...]:
        """Keys of :attr:`mobility`, ``"overall"`` first."""
        return ("overall",) + tuple(k for k in self.mobility if k != "overall")

    def _temperature_index(self, temperature: float) -> int:
        """Return the index of ``temperature`` in the file's temperature grid.

        Parameters
        ----------
        temperature : float
            Temperature in K.

        Returns
        -------
        int
            Position of the matching temperature.

        Raises
        ------
        ValueError
            If the file holds no data at that temperature. Mobility is not
            interpolated in temperature: AMSET is cheap to rerun at the cell
            temperature, and an interpolation across a sparse grid would hide
            the change of dominant scattering mechanism with T.
        """
        matches = np.flatnonzero(
            np.isclose(self.temperatures, temperature, rtol=_TEMPERATURE_RTOL, atol=0.0)
        )

        if matches.size == 0:
            available = ", ".join(f"{t:g}" for t in self.temperatures)

            raise ValueError(
                f"{self.filename.name} holds no data at T = {temperature:g} K"
                f" (available: {available} K). Rerun AMSET with this temperature"
                " in its 'temperatures' setting."
            )

        return int(matches[0])

    def _carrier_rows(self, carrier: str) -> NDArray:
        """Return the doping indices belonging to one carrier, by ascending |n|.

        Parameters
        ----------
        carrier : str
            ``"electrons"`` or ``"holes"``.

        Returns
        -------
        numpy.ndarray
            Integer indices into :attr:`doping`.

        Raises
        ------
        ValueError
            If ``carrier`` is not recognised or the file holds no data for it.
        """
        if carrier not in ("electrons", "holes"):
            raise ValueError(f"carrier must be 'electrons' or 'holes', not {carrier!r}")

        if carrier not in self.carriers:
            sign = "negative (n-type)" if carrier == "electrons" else "positive (p-type)"

            raise ValueError(
                f"{self.filename.name} holds no {_SINGULAR[carrier]} mobility: AMSET needs"
                f" {sign} doping values to compute it."
            )

        mask = self.doping < 0 if carrier == "electrons" else self.doping > 0
        rows = np.flatnonzero(mask)

        return rows[np.argsort(np.abs(self.doping[rows]))]

    def mobility_vs_doping(
            self, carrier: str, temperature: float = 300.0, mechanism: str = "overall"
    ) -> tuple[NDArray, NDArray]:
        """Return the isotropic mobility of one carrier against its concentration.

        The isotropic value is the mean of the tensor's eigenvalues (its trace
        over three), the same average AMSET prints in its own summary.

        Parameters
        ----------
        carrier : str
            ``"electrons"`` or ``"holes"``.
        temperature : float, optional
            Temperature in K; must be on the file's grid. Default is 300.
        mechanism : str, optional
            ``"overall"`` for the total mobility, or one key of
            :attr:`mobility` for the mobility limited by that mechanism alone.
            Default is ``"overall"``.

        Returns
        -------
        concentration : numpy.ndarray
            Carrier concentrations |n| in cm⁻³, ascending.
        mobility : numpy.ndarray
            Isotropic mobility in cm² V⁻¹ s⁻¹ at each concentration.

        Raises
        ------
        ValueError
            If the carrier, temperature or mechanism is not in the file.
        """
        tensors = self._tensors(carrier, temperature, mechanism)
        rows = self._carrier_rows(carrier)

        return np.abs(self.doping[rows]), np.trace(tensors, axis1=-2, axis2=-1) / 3

    def _tensors(self, carrier: str, temperature: float, mechanism: str) -> NDArray:
        """Return one carrier's mobility tensors at one temperature, by ascending |n|.

        Parameters
        ----------
        carrier : str
            ``"electrons"`` or ``"holes"``.
        temperature : float
            Temperature in K.
        mechanism : str
            Key of :attr:`mobility`.

        Returns
        -------
        numpy.ndarray
            Shape (n_carrier_doping, 3, 3).

        Raises
        ------
        ValueError
            If the mechanism is not in the file.
        """
        if mechanism not in self.mobility:
            raise ValueError(
                f"{self.filename.name} has no {mechanism!r} mobility (available:"
                f" {', '.join(self.mechanisms)})"
            )

        rows = self._carrier_rows(carrier)
        t = self._temperature_index(temperature)

        return self.mobility[mechanism][rows, t]

    def imp_share(self, carrier: str, temperature: float = 300.0) -> tuple[NDArray, NDArray] | None:
        """Return the ionized-impurity share of the total scattering.

        By Matthiessen's rule 1/μ = Σᵢ 1/μᵢ, so μ_overall / μ_IMP is the fraction
        of the total scattering rate that ionized impurities are responsible
        for. It measures how much the AMSET ``compensation_factor`` and
        ``defect_charge`` settings - neither stored in the transport file -
        matter at each doping.

        Parameters
        ----------
        carrier : str
            ``"electrons"`` or ``"holes"``.
        temperature : float, optional
            Temperature in K. Default is 300.

        Returns
        -------
        tuple of numpy.ndarray, or None
            Concentrations in cm⁻³ and the share at each, between 0 and 1; None
            if the file has no separate ``"IMP"`` mobility (AMSET was run with
            ``separate_mobility: false`` or without ionized impurity
            scattering).
        """
        if _IMP_LABEL not in self.mobility:
            return None

        n, total = self.mobility_vs_doping(carrier, temperature, "overall")
        _, imp = self.mobility_vs_doping(carrier, temperature, _IMP_LABEL)

        return n, total / imp

    def anisotropy(self, carrier: str, temperature: float = 300.0) -> tuple[NDArray, NDArray]:
        """Return the anisotropy of the total mobility tensor against concentration.

        Parameters
        ----------
        carrier : str
            ``"electrons"`` or ``"holes"``.
        temperature : float, optional
            Temperature in K. Default is 300.

        Returns
        -------
        concentration : numpy.ndarray
            Carrier concentrations |n| in cm⁻³, ascending.
        ratio : numpy.ndarray
            Largest over smallest eigenvalue of the mobility tensor; 1 for an
            isotropic material.
        """
        tensors = self._tensors(carrier, temperature, "overall")
        n, _ = self.mobility_vs_doping(carrier, temperature)

        # Mobility tensors are symmetric; AMSET writes them symmetrised.
        eigenvalues = np.linalg.eigvalsh(0.5 * (tensors + np.swapaxes(tensors, -1, -2)))

        return n, eigenvalues[..., -1] / eigenvalues[..., 0]


def _load_json(path: Path) -> dict:
    """Read a JSON transport file, gzipped or not.

    Parameters
    ----------
    path : Path
        The file to read.

    Returns
    -------
    dict
        The decoded JSON object.
    """
    opener = gzip.open if path.suffix == ".gz" else open

    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _from_json_dict(path: Path, data: dict) -> AmsetTransport:
    """Build an :class:`AmsetTransport` from AMSET's JSON layout.

    Parameters
    ----------
    path : Path
        The file the data came from, kept for messages.
    data : dict
        The decoded JSON.

    Returns
    -------
    AmsetTransport
        The parsed data.

    Raises
    ------
    ValueError
        If a required key is missing, or the file was written without
        mobilities.
    """
    missing = [k for k in ("doping", "temperatures", "fermi_levels") if k not in data]

    if missing:
        raise ValueError(
            f"{path.name} is not an AMSET transport file: missing {', '.join(missing)}"
        )

    if not data.get("mobility"):
        raise ValueError(
            f"{path.name} holds no mobility: AMSET was run with"
            " 'calculate_mobility: false', or the material is a metal."
        )

    doping = np.asarray(data["doping"], dtype=float)
    temperatures = np.asarray(data["temperatures"], dtype=float)
    mobility = {
        name: np.asarray(values, dtype=float).reshape(len(doping), len(temperatures), 3, 3)
        for name, values in data["mobility"].items()
    }

    return AmsetTransport(
        filename=path,
        doping=doping,
        temperatures=temperatures,
        fermi_levels=np.asarray(data["fermi_levels"], dtype=float).reshape(
            len(doping), len(temperatures)
        ),
        mobility=mobility,
    )


def _from_text(path: Path) -> AmsetTransport:
    """Build an :class:`AmsetTransport` from AMSET's ``txt``/``csv`` layout.

    AMSET writes one row per (doping, temperature) pair, doping-major, with a
    ``#`` header naming each column; tensors appear as their six upper-triangle
    components.

    Parameters
    ----------
    path : Path
        The file to read.

    Returns
    -------
    AmsetTransport
        The parsed data.

    Raises
    ------
    ValueError
        If the header is missing or names no mobility columns.
    """
    with open(path, encoding="utf-8") as handle:
        header = handle.readline()

    if not header.startswith("#"):
        raise ValueError(f"{path.name} has no '#' header line naming its columns")

    columns = header.lstrip("#").split()
    rows = np.atleast_2d(np.loadtxt(path))

    # Preserve the file's own ordering of doping and temperature, which
    # np.unique would sort.
    _, first = np.unique(rows[:, 0].round(), return_index=True)
    doping = rows[np.sort(first), 0].round()
    _, first_t = np.unique(rows[:, 1], return_index=True)
    temperatures = rows[np.sort(first_t), 1]

    shape = (len(doping), len(temperatures))
    fermi_levels = rows[:, 2].reshape(shape)

    mobility: dict[str, NDArray] = {}
    triu = np.triu_indices(3)

    for name in dict.fromkeys(c.split("_mobility_")[0] for c in columns if "_mobility_" in c):
        indices = [columns.index(f"{name}_mobility_{d}[cm^2/V.s]") for d in _TRIU_COMPONENTS]
        tensors = np.zeros(shape + (3, 3))
        tensors[..., triu[0], triu[1]] = rows[:, indices].reshape(shape + (6,))
        tensors[..., triu[1], triu[0]] = rows[:, indices].reshape(shape + (6,))
        mobility[name] = tensors

    if not mobility:
        raise ValueError(
            f"{path.name} holds no mobility columns: AMSET was run with"
            " 'calculate_mobility: false', or the material is a metal."
        )

    return AmsetTransport(
        filename=path,
        doping=doping,
        temperatures=temperatures,
        fermi_levels=fermi_levels,
        mobility=mobility,
    )


def load_amset_transport(filename: str | Path) -> AmsetTransport:
    """Read an AMSET transport file.

    Parameters
    ----------
    filename : str or Path
        A ``transport_<mesh>.json`` file (optionally gzipped), or the
        ``.txt``/``.csv`` file AMSET writes with ``file_format: txt``.

    Returns
    -------
    AmsetTransport
        The doping, temperatures, Fermi levels and mobility tensors.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If it is not an AMSET transport file, or holds no mobility.
    """
    path = Path(filename)

    if not path.is_file():
        raise FileNotFoundError(f"No AMSET transport file at {path}")

    suffixes = "".join(path.suffixes[-2:])

    if path.suffix == ".json" or suffixes == ".json.gz":
        return _from_json_dict(path, _load_json(path))

    return _from_text(path)


def _loglog_interpolator(
        concentration: NDArray, mobility: NDArray
) -> Callable[[NDArray], NDArray]:
    """Return a function interpolating log μ linearly in log n, flat outside the grid.

    Parameters
    ----------
    concentration : numpy.ndarray
        Ascending concentrations in cm⁻³.
    mobility : numpy.ndarray
        Mobility at each concentration in cm² V⁻¹ s⁻¹.

    Returns
    -------
    callable
        Maps concentrations to mobilities. Outside the grid it returns the
        nearest end value; range policy is the caller's job.
    """
    log_n = np.log10(concentration)
    log_mu = np.log10(mobility)

    def interpolate(n: NDArray) -> NDArray:
        return 10.0 ** np.interp(np.log10(n), log_n, log_mu)

    return interpolate


@dataclass(frozen=True)
class MobilityModel:
    """A doping-dependent mobility μ(n), ready to be passed to Γₚᵥ.

    Instances are made by :func:`select_mobility` and are callable: ``model(n)``
    returns the mobility in cm² V⁻¹ s⁻¹ at doping density ``n`` in cm⁻³.
    ``str(model)`` is the summary :func:`select_mobility` prints.

    Attributes
    ----------
    filename : Path
        The AMSET transport file the data came from.
    carrier : str
        The carrier choice: ``"limiting"``, ``"electrons"`` or ``"holes"``.
    mechanism : str
        The mobility used: ``"overall"``, or a single scattering mechanism.
    temperature : float
        Temperature in K.
    curves : dict of str to tuple of numpy.ndarray
        For each carrier taken into account, its computed concentrations
        (cm⁻³, ascending) and isotropic mobilities (cm² V⁻¹ s⁻¹).
    lower_bound : float
        Concentration below which μ is held at its lowest-doping value.
    upper_bound : float
        Highest concentration the AMSET grid reaches for every carrier taken
        into account.
    allow_out_of_range : bool
        Whether a concentration above :attr:`upper_bound` is held at the
        highest-doping value with a warning, instead of refused.
    notes : tuple of str
        The checks' findings, in the order the summary prints them.
    """

    filename: Path
    carrier: str
    mechanism: str
    temperature: float
    curves: dict[str, tuple[NDArray, NDArray]] = field(repr=False)
    lower_bound: float
    upper_bound: float
    allow_out_of_range: bool = False
    notes: tuple[str, ...] = ()

    def by_carrier(self, n: ArrayLike) -> dict[str, NDArray]:
        """Return each carrier's mobility at the given concentrations.

        Parameters
        ----------
        n : array_like
            Doping densities in cm⁻³ (positive).

        Returns
        -------
        dict of str to numpy.ndarray
            Mobility in cm² V⁻¹ s⁻¹ per carrier taken into account.

        Raises
        ------
        ValueError
            If a concentration exceeds :attr:`upper_bound` and
            :attr:`allow_out_of_range` is False.
        """
        n = np.asarray(n, dtype=float)

        if np.any(n <= 0):
            raise ValueError("Doping densities must be positive (|N_D - N_A| in cm⁻³)")

        above = n > self.upper_bound * (1 + 1e-9)

        if np.any(above):
            message = (
                f"n = {np.max(n):.3g} cm⁻³ is above the highest doping in"
                f" {self.filename.name} ({self.upper_bound:.3g} cm⁻³)"
            )

            if not self.allow_out_of_range:
                raise ValueError(
                    f"{message}. Extend AMSET's 'doping' list, or pass"
                    " allow_out_of_range=True to hold μ at its last value."
                )

            warnings.warn(f"{message}; μ held at its last value.", UserWarning, stacklevel=3)

        return {
            carrier: _loglog_interpolator(*curve)(n) for carrier, curve in self.curves.items()
        }

    def limiting_carrier(self, n: ArrayLike) -> NDArray:
        """Return which carrier sets the mobility at each concentration.

        Parameters
        ----------
        n : array_like
            Doping densities in cm⁻³.

        Returns
        -------
        numpy.ndarray of str
            ``"electrons"`` or ``"holes"`` per concentration.
        """
        per_carrier = self.by_carrier(n)
        names = list(per_carrier)
        stacked = np.stack([per_carrier[c] for c in names])

        return np.asarray(names)[np.argmin(stacked, axis=0)]

    @overload
    def __call__(self, n: float) -> float: ...

    @overload
    def __call__(self, n: NDArray) -> NDArray: ...

    def __call__(self, n: ArrayLike) -> NDArray | float:
        """Return the mobility in cm² V⁻¹ s⁻¹ at doping density ``n`` in cm⁻³.

        Parameters
        ----------
        n : array_like
            Doping densities in cm⁻³.

        Returns
        -------
        float or numpy.ndarray
            The lower of the carriers taken into account, at each density; a
            float for scalar input.
        """
        per_carrier = self.by_carrier(n)
        mu = np.min(np.stack(list(per_carrier.values())), axis=0)

        return float(mu) if mu.ndim == 0 else mu

    def __str__(self) -> str:
        """Return the summary of what mobility is being used, and why."""
        carriers = " and ".join(self.curves)

        if self.carrier == "limiting" and len(self.curves) == 2:
            choice = "limiting carrier: the lower of the electron and hole mobilities"
        elif self.carrier == "limiting":
            choice = f"limiting carrier requested, but only {carriers} in the file"
        else:
            choice = f"{_SINGULAR[self.carrier]} mobility (chosen)"

        lines = [
            f"AMSET transport: {self.filename.name}  (T = {self.temperature:g} K)",
            f"Mobility used for Γₚᵥ: {choice}",
            f"Scattering: '{self.mechanism}'"
            + (" (total)" if self.mechanism == "overall" else " only"),
            "Tensor: isotropic average (trace / 3)",
            f"Doping grid: {self.lower_bound:.3g} - {self.upper_bound:.3g} cm⁻³",
        ]
        lines.extend(f"  - {note}" for note in self.notes)

        return "\n".join(lines)


def _percent(share: float) -> str:
    """Format a fraction as a short percentage."""
    return f"{100 * share:.3g}%"


def select_mobility(
        transport: AmsetTransport | str | Path,
        carrier: str = "limiting",
        mechanism: str = "overall",
        temperature: float = 300.0,
        dop_range: tuple[float, float] = _DEFAULT_DOP_RANGE,
        *,
        allow_out_of_range: bool = False,
        verbose: bool = True,
) -> MobilityModel:
    """Choose the mobility Γₚᵥ will use from an AMSET transport file, and check it.

    Γₚᵥ assumes one mobility for both carriers (μₙ = μₚ, Crovetto 2024 section
    2). Equation S2 of that paper's supplementary material proposes how to
    reduce two mobilities to one: the minority-carrier mobility under low
    injection (Δn ≪ n), and min(μₙ, μₚ) otherwise, noting that the rule has
    not yet been tested against drift-diffusion simulations. The default
    ``carrier="limiting"`` is the second branch, the lower of the electron and
    hole mobilities at each doping density: conservative, and it needs no
    absorber type. For an absorber known to operate in low injection, pass the
    minority carrier instead - ``"holes"`` for an n-type absorber,
    ``"electrons"`` for a p-type one.

    The doping density Γₚᵥ is evaluated at is the majority-carrier
    concentration, and AMSET computes each carrier's mobility as the majority
    carrier at that concentration. Using a hole mobility at the same |n| for an
    n-type absorber therefore approximates the minority-carrier mobility by the
    majority one at the same ionized-impurity density.

    The following are checked, and every finding is reported in the summary:

    * a carrier missing from the file (only one sign of doping computed),
      which leaves μₙ = μₚ resting on one carrier: ``UserWarning``;
    * a single scattering mechanism chosen instead of ``"overall"``:
      ``UserWarning``;
    * the AMSET doping grid not covering ``dop_range``. Below the grid, μ is
      held at its lowest-doping value; that is exact when ionized impurities
      are negligible there (the mobility is phonon-limited and flat), which is
      only noted, and conservative otherwise, which is a ``UserWarning``.
      Above the grid, :class:`MobilityModel` refuses unless
      ``allow_out_of_range``;
    * ionized impurities above 10% of the scattering inside ``dop_range``,
      where AMSET's ``compensation_factor`` and ``defect_charge`` (not stored
      in the file) matter: ``UserWarning``;
    * a mobility tensor whose eigenvalues differ by more than 1.5x, which the
      isotropic average Γₚᵥ needs hides: ``UserWarning``.

    Parameters
    ----------
    transport : AmsetTransport, str or Path
        Data from :func:`load_amset_transport`, or the path to the file.
    carrier : str, optional
        ``"limiting"`` (default), ``"electrons"`` or ``"holes"``.
    mechanism : str, optional
        ``"overall"`` (default) for the total mobility, or one scattering
        mechanism in the file (e.g. ``"POP"``) to see the mobility it alone
        would allow.
    temperature : float, optional
        Temperature in K, which must be on the file's grid; use the cell
        temperature ``Tcell``. Default is 300.
    dop_range : tuple of float, optional
        Doping densities in cm⁻³ the model will be evaluated over, used for
        the coverage and impurity checks. Default is the Crovetto 2024 table 1
        range, 10¹⁰ - 10¹⁸.
    allow_out_of_range : bool, optional
        If True, a density above the AMSET grid is held at the highest-doping
        mobility with a warning, instead of raising. Default is False.
        Keyword-only.
    verbose : bool, optional
        Print the summary. Default is True. Keyword-only.

    Returns
    -------
    MobilityModel
        Callable μ(n) in cm² V⁻¹ s⁻¹, with the choices and checks recorded.

    Raises
    ------
    ValueError
        If ``carrier`` or ``mechanism`` is not recognised or not in the file,
        the temperature is not on the file's grid, or (without
        ``allow_out_of_range``) ``dop_range`` reaches above the AMSET grid.
    """
    if not isinstance(transport, AmsetTransport):
        transport = load_amset_transport(transport)

    if carrier not in CARRIER_CHOICES:
        raise ValueError(f"carrier must be one of {CARRIER_CHOICES}, not {carrier!r}")

    if mechanism not in transport.mobility:
        raise ValueError(
            f"{transport.filename.name} has no {mechanism!r} mobility (available:"
            f" {', '.join(transport.mechanisms)})"
        )

    name = transport.filename.name
    notes: list[str] = []

    def report(message: str, warn: bool = True) -> None:
        notes.append(("⚠ " if warn else "ℹ ") + message)

        if warn:
            warnings.warn(f"{name}: {message}", UserWarning, stacklevel=3)

    # --- which carriers ---------------------------------------------------
    if carrier == "limiting":
        used = list(transport.carriers)

        if len(used) == 1:
            other = "holes" if used[0] == "electrons" else "electrons"
            sign = "positive (p-type)" if other == "holes" else "negative (n-type)"
            report(
                f"Only {_SINGULAR[used[0]]} mobility in the file, so it is used for"
                f" both carriers (Γₚᵥ assumes μn = μp). The {_SINGULAR[other]} mobility"
                " is unknown;"
                f" rerun AMSET with {sign} doping values to include it."
            )
    else:
        used = [carrier]

    curves = {c: transport.mobility_vs_doping(c, temperature, mechanism) for c in used}

    if mechanism != "overall":
        report(
            f"Using the '{mechanism}'-limited mobility, not the total: it ignores"
            " every other scattering mechanism and overestimates μ. Meant for"
            " diagnostics."
        )

    # --- doping coverage ----------------------------------------------------
    lower_bound = max(curve[0][0] for curve in curves.values())
    upper_bound = min(curve[0][-1] for curve in curves.values())
    lo, hi = dop_range

    if lo < lower_bound:
        shares = [transport.imp_share(c, temperature) for c in used]

        if any(s is None for s in shares):
            report(
                f"The AMSET grid starts at {lower_bound:.3g} cm⁻³, above the"
                f" {lo:.3g} cm⁻³ requested: μ is held at its {lower_bound:.3g} value"
                " below that. The file has no separate IMP mobility, so whether μ is"
                " already flat (phonon-limited) there cannot be checked."
            )
        else:
            share_at_bottom = max(
                float(np.interp(np.log10(lower_bound), np.log10(s[0]), s[1]))
                for s in shares
                if s is not None
            )

            if share_at_bottom < _FLAT_EXTRAPOLATION_IMP_SHARE:
                report(
                    f"The AMSET grid starts at {lower_bound:.3g} cm⁻³, above the"
                    f" {lo:.3g} cm⁻³ requested: μ is held at its {lower_bound:.3g}"
                    " value below that. Ionized impurities are only"
                    f" {_percent(share_at_bottom)} of the scattering there, so μ is"
                    " phonon-limited and this is accurate.",
                    warn=False,
                )
            else:
                report(
                    f"The AMSET grid starts at {lower_bound:.3g} cm⁻³, above the"
                    f" {lo:.3g} cm⁻³ requested: μ is held at its {lower_bound:.3g}"
                    " value below that, but ionized impurities are still"
                    f" {_percent(share_at_bottom)} of the scattering there, so μ is"
                    " underestimated at lower doping. Extend AMSET's 'doping' list"
                    f" down to {lo:.3g}."
                )

    if hi > upper_bound:
        message = (
            f"The AMSET grid ends at {upper_bound:.3g} cm⁻³, below the {hi:.3g}"
            " cm⁻³ requested."
        )

        if not allow_out_of_range:
            raise ValueError(
                f"{name}: {message} Extend AMSET's 'doping' list, or pass"
                " allow_out_of_range=True to hold μ at its last value."
            )

        report(f"{message} μ is held at its last value above it.")

    # --- impurity sensitivity and anisotropy, inside the requested window ---
    for c in used:
        share = transport.imp_share(c, temperature)

        if share is not None:
            inside = (share[0] >= lo) & (share[0] <= hi)
            significant = inside & (share[1] > _SIGNIFICANT_IMP_SHARE)

            if np.any(significant):
                first = share[0][significant][0]
                report(
                    f"Ionized impurities exceed {_percent(_SIGNIFICANT_IMP_SHARE)} of"
                    f" {_SINGULAR[c]} scattering from {first:.3g} cm⁻³"
                    f" ({_percent(share[1][significant][0])} there). The"
                    " compensation_factor and defect_charge AMSET used (not stored"
                    " in the transport file) matter from that doping on."
                )

        n, ratio = transport.anisotropy(c, temperature)
        inside = (n >= lo) & (n <= hi)

        if np.any(inside) and np.max(ratio[inside]) > _ANISOTROPY_RATIO:
            report(
                f"The {_SINGULAR[c]} mobility is anisotropic (eigenvalues differ by up to"
                f" {np.max(ratio[inside]):.2g}x); Γₚᵥ takes the isotropic average,"
                " which a thin film along one direction may not see."
            )

    model = MobilityModel(
        filename=transport.filename,
        carrier=carrier,
        mechanism=mechanism,
        temperature=temperature,
        curves=curves,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        allow_out_of_range=allow_out_of_range,
        notes=tuple(notes),
    )

    if carrier == "limiting" and len(used) == 2:
        grid = np.unique(np.concatenate([curve[0] for curve in curves.values()]))
        grid = grid[(grid >= lo) & (grid <= min(hi, upper_bound))]

        if grid.size:
            limiting = model.limiting_carrier(grid)
            changes = [
                f"{limiting[i]} from {grid[i]:.3g} cm⁻³"
                for i in range(len(grid))
                if i == 0 or limiting[i] != limiting[i - 1]
            ]
            model = replace(
                model, notes=model.notes + ("Limiting carrier: " + "; ".join(changes) + ".",)
            )

    if verbose:
        print(model)

    return model


# Fixed colour per scattering mechanism, so a mechanism keeps its colour across
# materials and figures whatever subset a file holds.
_MECHANISM_COLOURS = {
    "ADP": "#2a78d6",
    "IMP": "#eb6834",
    "POP": "#1baf7a",
    "PIE": "#6250d6",
    "CRT": "#e87ba4",
    "MRT": "#008300",
}
_FALLBACK_COLOUR = "#8a8984"
_OVERALL_COLOUR = "#0b0b0b"


def plot_mobility(
        transport: AmsetTransport | str | Path,
        temperature: float = 300.0,
        ax: Axes | None = None,
        dop_range: tuple[float, float] | None = _DEFAULT_DOP_RANGE,
) -> Axes:
    """Plot mobility against carrier concentration, total and per scattering mechanism.

    Electrons are drawn as solid lines and holes dashed; each mechanism keeps a
    fixed colour and the total is black. The doping window Γₚᵥ was fitted over
    is shaded, so it is visible at a glance which mechanisms matter where the
    figure of merit is evaluated.

    Parameters
    ----------
    transport : AmsetTransport, str or Path
        Data from :func:`load_amset_transport`, or the path to the file.
    temperature : float, optional
        Temperature in K. Default is 300.
    ax : Axes or None, optional
        Axes to draw on; a new figure is made if None.
    dop_range : tuple of float or None, optional
        Doping window to shade, in cm⁻³. Default is the Crovetto 2024 table 1
        range; None shades nothing.

    Returns
    -------
    Axes
        The axes drawn on.
    """
    if not isinstance(transport, AmsetTransport):
        transport = load_amset_transport(transport)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4), dpi=120)

    styles = {"electrons": "-", "holes": "--"}

    for carrier in transport.carriers:
        for mechanism in transport.mechanisms:
            n, mu = transport.mobility_vs_doping(carrier, temperature, mechanism)
            overall = mechanism == "overall"
            label = "total" if overall else mechanism

            if len(transport.carriers) > 1:
                label = f"{label} ({carrier})"

            ax.plot(
                n,
                mu,
                styles[carrier],
                color=_OVERALL_COLOUR if overall else _MECHANISM_COLOURS.get(
                    mechanism, _FALLBACK_COLOUR
                ),
                linewidth=2.5 if overall else 1.5,
                marker="o",
                markersize=4 if overall else 3,
                label=label,
                zorder=3 if overall else 2,
            )

    if dop_range is not None:
        ax.axvspan(*dop_range, color="#d9d8d3", alpha=0.5, linewidth=0, zorder=0,
                   label="Γₚᵥ doping range")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Carrier concentration (cm$^{-3}$)")
    ax.set_ylabel("Mobility (cm$^2$V$^{-1}$s$^{-1}$)")
    ax.set_title(f"AMSET mobility, T = {temperature:g} K")
    ax.grid(True, which="major", color="#e4e3df", linewidth=0.6, zorder=1)
    # Outside the axes: with both carriers there are up to nine entries, and
    # every corner of a mobility plot has lines running through it.
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    return ax
