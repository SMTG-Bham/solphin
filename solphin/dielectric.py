r"""Ionic contribution to the static dielectric constant, from VASP or CASTEP output.

The static dielectric constant has two parts: the ion-clamped electronic
response :math:`\varepsilon_\infty`, which :func:`solphin.optics.calc_dielectric`
returns, and the response of the lattice to the field, which this module
returns. The total static constant is their sum.
"""

import gzip
import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# The VASP OUTCAR header, and the left-hand header of the CASTEP table. Both
# are matched case-insensitively: CASTEP's capitalisation has shifted between
# releases, and the VASP line is padded differently by different builds.
_VASP_IONIC_HEADER = re.compile(
    r"MACROSCOPIC STATIC DIELECTRIC TENSOR IONIC CONTRIBUTION", re.IGNORECASE
)
_CASTEP_PERMITTIVITY_HEADER = re.compile(r"Optical Permittivity", re.IGNORECASE)


def _read_lines(path: Path) -> list[str]:
    """Read a text output file, transparently decompressing a gzipped one.

    OUTCARs are routinely gzipped to keep an archive of calculations to a
    sensible size, and pymatgen reads them either way, so these readers do
    too.

    Parameters
    ----------
    path : Path
        File to read. A ``.gz`` suffix selects decompression.

    Returns
    -------
    list of str
        The file's lines, without their terminators.
    """
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            return handle.read().splitlines()
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _collect_float_rows(
        lines: list[str],
        start: int,
        n_rows: int,
        widths: tuple[int, ...],
        path: Path,
        block: str,
) -> list[list[float]]:
    """Read the numeric rows of a tensor block following a header line.

    Scans forward from the header, skipping blank lines and the rule of
    dashes that underlines it, and stops at the first line that is not a row
    of ``widths`` numbers - a rule of equals signs, a fresh header, or the
    end of the block.

    Parameters
    ----------
    lines : list of str
        The file's lines.
    start : int
        Index of the header line; scanning begins at the line after it.
    n_rows : int
        Number of numeric rows the block should contain.
    widths : tuple of int
        Acceptable column counts. A numeric row of any other width ends the
        block rather than being read as part of it.
    path : Path
        File being read, named in error messages.
    block : str
        Human-readable block name for error messages.

    Returns
    -------
    list of list of float
        The parsed rows, all of the same width.

    Raises
    ------
    ValueError
        If fewer than ``n_rows`` rows of consistent width were found.
    """
    rows: list[list[float]] = []

    for raw in lines[start + 1:]:
        stripped = raw.strip()
        if not stripped or set(stripped) <= set("- "):
            continue
        try:
            row = [float(token) for token in stripped.split()]
        except ValueError:
            break
        if len(row) not in widths or (rows and len(row) != len(rows[0])):
            break
        rows.append(row)
        if len(rows) == n_rows:
            break

    if len(rows) != n_rows:
        raise ValueError(
            f"Expected {n_rows} rows of numbers under the {block} block in {path},"
            f" found {len(rows)}."
        )
    return rows


def _parse_ionic_dielectric_vasp(filename: str | Path) -> tuple[float, NDArray]:
    """Read the ionic dielectric tensor from a VASP OUTCAR.

    Parameters
    ----------
    filename : str or Path
        Path of the OUTCAR file.

    Returns
    -------
    eps_ion : float
        Ionic contribution averaged over the tensor diagonal.
    eps_ion_tensor : numpy.ndarray
        Ionic contribution to the dielectric tensor, shape (3, 3).

    Raises
    ------
    ValueError
        If the OUTCAR holds no ionic dielectric block, which means the run
        did not compute one.
    """
    path = Path(filename)
    lines = _read_lines(path)

    starts = [i for i, line in enumerate(lines) if _VASP_IONIC_HEADER.search(line)]
    if not starts:
        raise ValueError(
            f"No ionic dielectric tensor in {path}: the OUTCAR has no"
            " 'MACROSCOPIC STATIC DIELECTRIC TENSOR IONIC CONTRIBUTION' block."
            " Computing one needs ionic displacements (IBRION=6 or 8) together"
            " with LEPSILON=.TRUE. (the 'dfpt' patch) or LCALCEPS=.TRUE. (the"
            " 'lattice_response' patch)."
        )

    rows = _collect_float_rows(
        lines, starts[-1], 3, (3,), path, "ionic dielectric tensor"
    )
    eps_ion_tensor = np.array(rows, dtype=float)

    return float(np.mean(eps_ion_tensor.diagonal())), eps_ion_tensor


def _read_castep_permittivity(filename: str | Path) -> tuple[NDArray, NDArray]:
    r"""Read the optical and DC permittivity tensors from a ``.castep`` file.

    A ``task : Efield`` run prints the two tensors side by side, six numbers
    per row - the optical (:math:`f\to\infty`) tensor on the left and the DC
    (:math:`f=0`) tensor on the right::

        Optical Permittivity (f->infinity)             DC Permittivity (f=0)
        ----------------------------------             ---------------------
         4.50788     0.00000     0.00000         6.64363     0.00000     0.00000
         0.00000     4.50788     0.00000         0.00000     6.64363     0.00000
         0.00000     0.00000     4.63846         0.00000     0.00000     7.15660

    With ``efield_calc_ion_permittivity : false``, or for a bare ``task :
    Phonon`` run, only the three left-hand columns are printed.

    Parameters
    ----------
    filename : str or Path
        Path of the ``<seed>.castep`` output file.

    Returns
    -------
    optical : numpy.ndarray
        Optical (ion-clamped) permittivity tensor, shape (3, 3).
    dc : numpy.ndarray
        DC (static) permittivity tensor, shape (3, 3).

    Raises
    ------
    ValueError
        If the file holds no permittivity table, or holds one without the DC
        columns.
    """
    path = Path(filename)
    lines = _read_lines(path)

    starts = [
        i for i, line in enumerate(lines) if _CASTEP_PERMITTIVITY_HEADER.search(line)
    ]
    if not starts:
        raise ValueError(
            f"No permittivity table in {path}: the run printed no"
            " 'Optical Permittivity' block. Computing one needs an electric-field"
            " response calculation - 'task : Efield', which solphin writes from"
            " the 'dfpt' patch."
        )

    # CASTEP appends to <seed>.castep on a continuation, so the last table is
    # the current one.
    rows = _collect_float_rows(lines, starts[-1], 3, (3, 6), path, "permittivity")

    if len(rows[0]) == 3:
        raise ValueError(
            f"The permittivity table in {path} has no DC (f=0) columns, so the"
            " ionic contribution cannot be recovered from it. Rerun with"
            " 'efield_calc_ion_permittivity : true', which solphin sets in the"
            " 'dfpt' patch."
        )

    table = np.array(rows, dtype=float)
    return table[:, :3], table[:, 3:]


def _parse_ionic_dielectric_castep(filename: str | Path) -> tuple[float, NDArray]:
    """Derive the ionic dielectric tensor from a CASTEP permittivity table.

    The ionic contribution is the DC permittivity less the optical one, both
    of which ``task : Efield`` prints.

    Parameters
    ----------
    filename : str or Path
        Path of the ``<seed>.castep`` output file.

    Returns
    -------
    eps_ion : float
        Ionic contribution averaged over the tensor diagonal.
    eps_ion_tensor : numpy.ndarray
        Ionic contribution to the dielectric tensor, shape (3, 3).
    """
    optical, dc = _read_castep_permittivity(filename)
    eps_ion_tensor = dc - optical

    return float(np.mean(eps_ion_tensor.diagonal())), eps_ion_tensor


def _find_castep_file(directory: str | Path, seedname: str | None) -> Path:
    """Locate the ``.castep`` output file inside a calculation directory.

    Parameters
    ----------
    directory : str or Path
        Directory containing the CASTEP output.
    seedname : str or None
        CASTEP seed. If given, ``<seedname>.castep`` is required; if None,
        the directory must hold exactly one ``*.castep`` file.

    Returns
    -------
    Path
        Path of the ``.castep`` file.

    Raises
    ------
    FileNotFoundError
        If the named or globbed ``.castep`` file does not exist.
    ValueError
        If no seedname was given and several ``.castep`` files match.
    """
    directory = Path(directory)

    if seedname is not None:
        path = directory / f"{seedname}.castep"
        if not path.is_file():
            raise FileNotFoundError(f"No CASTEP output file at {path}")
        return path

    candidates = sorted(directory.glob("*.castep"))
    if not candidates:
        raise FileNotFoundError(f"No *.castep file found in {directory}")
    if len(candidates) > 1:
        names = ", ".join(candidate.name for candidate in candidates)
        raise ValueError(
            f"Several CASTEP output files in {directory}: {names};"
            " pass seedname to choose one."
        )
    return candidates[0]


def _resolve_ionic_dielectric_file(
        filename: str | Path, code: str, seedname: str | None
) -> Path:
    """Resolve the output file to parse, accepting a directory or a file.

    Parameters
    ----------
    filename : str or Path
        Either the output file itself, or the calculation directory holding
        it.
    code : str
        Which first-principles code produced the data, ``"vasp"`` or
        ``"castep"``.
    seedname : str or None
        CASTEP seed used to disambiguate the ``.castep`` file; ignored for
        VASP.

    Returns
    -------
    Path
        Path of the file to parse.

    Raises
    ------
    ValueError
        If ``code`` is not ``"vasp"`` or ``"castep"``.
    """
    if code not in ("vasp", "castep"):
        raise ValueError(f"Unsupported code {code!r}; expected 'vasp' or 'castep'.")

    path = Path(filename)
    if not path.is_dir():
        return path

    if code == "vasp":
        # An archived calculation usually keeps the OUTCAR gzipped. Falling
        # back to the plain name when neither exists keeps the resulting
        # FileNotFoundError pointing at the file the caller expected.
        outcar = path / "OUTCAR"
        gzipped = path / "OUTCAR.gz"
        return gzipped if gzipped.is_file() and not outcar.is_file() else outcar
    return _find_castep_file(path, seedname)


def parse_ionic_dielectric(
        filename: str | Path, code: str = "vasp", seedname: str | None = None
) -> tuple[float, NDArray]:
    """Read the ionic contribution to the static dielectric constant.

    This is the lattice part only. The total static dielectric constant is
    this plus the electronic part ``eps_inf`` from
    :func:`solphin.optics.calc_dielectric`::

        eps_inf, *_ = optics.calc_dielectric("OPT/vasprun.xml")
        eps_ion, _ = dielectric.parse_ionic_dielectric("DFPT/OUTCAR")
        eps_static = eps_inf + eps_ion

    The result is only as good as the geometry it was computed at: residual
    forces produce imaginary modes and a meaningless tensor, so the structure
    must be tightly relaxed in the same computational regime first.

    Parameters
    ----------
    filename : str or Path
        The output file to parse - ``OUTCAR`` for VASP, ``<seed>.castep`` for
        CASTEP - or the directory holding it. A gzipped file is read as-is,
        and a VASP directory holding only ``OUTCAR.gz`` resolves to that.
    code : str, optional
        Which code produced the output, ``"vasp"`` or ``"castep"``. Default
        is ``"vasp"``.
    seedname : str or None, optional
        CASTEP seed, used only when ``filename`` is a directory holding more
        than one ``.castep`` file. Ignored for VASP. Default is None.

    Returns
    -------
    eps_ion : float
        Ionic contribution to the static dielectric constant, averaged over
        the tensor diagonal.
    eps_ion_tensor : numpy.ndarray
        Ionic contribution to the dielectric tensor, shape (3, 3).

    Raises
    ------
    ValueError
        If ``code`` is not ``"vasp"`` or ``"castep"``, or if the output holds
        no ionic dielectric data.
    """
    path = _resolve_ionic_dielectric_file(filename, code, seedname)

    if code == "castep":
        return _parse_ionic_dielectric_castep(path)
    return _parse_ionic_dielectric_vasp(path)
