"""Band-structure workflows: k-path generation, calculation inputs, reconstruction and plotting.

The k-path generation and plotting are code-agnostic. VASP calculations are
assembled from KPOINTS files and read back from ``vasprun.xml``; CASTEP
calculations restart from the SCF ``.check`` file with the band path appended
to a copy of the ``.cell``, and are read back from ``<seed>.bands``.
"""

import math
import re
import shutil
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.outputs import BSVasprun
from sumo.electronic_structure.bandstructure import get_reconstructed_band_structure
from sumo.electronic_structure.dos import load_dos
from sumo.io.castep import band_structure as castep_band_structure
from sumo.io.castep import copy_param
from sumo.io.castep import read_dos as castep_read_dos
from sumo.io.castep import write_kpoint_files as castep_write_kpoint_files
from sumo.plotting.bs_plotter import SBSPlotter
from sumo.plotting.dos_plotter import SDOSPlotter
from sumo.symmetry.kpoints import get_path_data

from solphin.vasp_inputs import write_vasp_calculation

# The kpoint path/list block sumo's cell-file label reader requires; a .cell
# without one makes that reader call sys.exit, so it is checked up front.
_CASTEP_KPOINT_BLOCK_RE = re.compile(
    r"^\s*%block\s+(bs|spectral)_kpoint(s)?_(path|list)",
    re.IGNORECASE | re.MULTILINE,
)


def generate_band_structure_path(
        structure: Structure,
        definition: str = "bradcrack",
        symprec: float = 0.01,
        density: int = 60,
        cartesian: bool = False
) -> tuple[Structure, tuple[list[NDArray], list[str]]]:
    """Generate a high-symmetry k-point path for band structure calculations.

    If the input structure differs from the canonical primitive cell, the
    path is recomputed with the primitive structure for consistency.

    Parameters
    ----------
    structure : Structure
        Input crystal structure.
    definition : str, optional
        K-path generation scheme. Default is ``"bradcrack"``.
    symprec : float, optional
        Symmetry tolerance used for structure analysis. Default is ``0.01``.
    density : int, optional
        Number of k-points per unit length along the path; higher values
        produce smoother band structures. Default is ``60``.
    cartesian : bool, optional
        Return k-points in Cartesian rather than reciprocal coordinates.
        Default is ``False``.

    Returns
    -------
    canonical_structure : Structure
        Primitive/canonical structure used for the k-path generation.
    kpath : tuple of (list of numpy.ndarray, list of str)
        The k-point path coordinates and the corresponding high-symmetry
        labels.
    """
    kpath, kpoints, labels = get_path_data(
        structure,
        mode=definition,
        symprec=symprec,
        kpt_list=None,
        labels=None,
        spg=None,
        line_density=density,
        cart_coords=cartesian,
    )

    if not np.allclose(structure.lattice.matrix, kpath.prim.lattice.matrix):

        canonical_structure = kpath.prim

        _, canonical_kpoints, canonical_labels = get_path_data(
            canonical_structure,
            mode=definition,
            symprec=symprec,
            kpt_list=None,
            labels=None,
            spg=None,
            line_density=density,
            cart_coords=cartesian,
        )

        print("INFO: The canonical structure differs from the supplied structure.")

    else:
        canonical_structure = structure
        canonical_kpoints = kpoints
        canonical_labels = labels

    print(f"Generated high-symmetry path of {len(canonical_kpoints)} k-points")

    return canonical_structure, (canonical_kpoints, canonical_labels)


# Simplified version of sumo.io.vasp.write_kpoint_files
def _write_kpoint_files(
        directory: str | Path,
        kpoints: list[NDArray],
        labels: list[str],
        make_folders: bool = True,
        ibzkpt: Kpoints | None = None,
        kpts_per_split: int | None = None,
        cart_coords: bool = False,
) -> list[str]:
    """Generate and write KPOINTS files for band structure calculations.

    The band path can be split into segments for parallel or chunked runs,
    written either into separate folders or as individual files. Hybrid
    calculations are supported by prepending an irreducible k-point mesh
    with adjusted weights.

    Parameters
    ----------
    directory : str or Path
        Output directory where KPOINTS files or folders are written.
    kpoints : list of numpy.ndarray
        K-point coordinates defining the band path.
    labels : list of str
        High-symmetry point labels corresponding to ``kpoints``.
    make_folders : bool, optional
        Create a separate folder for each split segment. Default is ``True``.
    ibzkpt : Kpoints or None, optional
        Irreducible Brillouin-zone k-point mesh for hybrid calculations.
        If provided, k-point weights are set accordingly. Default is None.
    kpts_per_split : int or None, optional
        Number of k-points per split segment. Default is None, meaning no
        splitting.
    cart_coords : bool, optional
        Treat ``kpoints`` as Cartesian rather than reciprocal coordinates.
        Default is ``False``.

    Returns
    -------
    list of str
        Folder name for each generated KPOINTS segment, or empty strings
        when folders are not used.
    """
    if kpts_per_split:
        kpt_splits = [
            kpoints[i: i + kpts_per_split]
            for i in range(0, len(kpoints), kpts_per_split)
        ]
        label_splits = [
            labels[i: i + kpts_per_split]
            for i in range(0, len(labels), kpts_per_split)
        ]
    else:
        kpt_splits = [kpoints]
        label_splits = [labels]

    if cart_coords:
        coord_type = "cartesian"
        style = Kpoints.supported_modes.Cartesian
    else:
        coord_type = "reciprocal"
        style = Kpoints.supported_modes.Reciprocal

    kpt_files = []
    for kpt_split, label_split in zip(kpt_splits, label_splits):
        if ibzkpt is not None:
            # hybrid calculation so set k-point weights to 0
            kpt_weights = ibzkpt.kpts_weights + [0] * len(kpt_split)
            kpt_split = ibzkpt.kpts + kpt_split
            label_split = [""] * len(ibzkpt.kpts) + label_split
        else:
            # non-SCF calculation so set k-point weights to 1
            kpt_weights = [1] * len(kpt_split)

        segment = " -> ".join([label for label in label_split if label])
        kpt_file = Kpoints(
            comment=segment,
            num_kpts=len(kpt_split),
            kpts=kpt_split,
            kpts_weights=kpt_weights,
            style=style,
            coord_type=coord_type,
            labels=label_split,
        )
        kpt_files.append(kpt_file)

    pad = int(math.floor(math.log10(len(kpt_files)))) + 2

    folders = []

    if make_folders:
        for i, kpt_file in enumerate(kpt_files):
            folder = f"split-{str(i + 1).zfill(pad)}"
            folders.append(folder)

            folder_path = Path(directory) / folder if directory else Path(folder)
            folder_path.mkdir(parents=True, exist_ok=True)

            kpt_file.write_file(folder_path / "KPOINTS")

    else:
        folders.append("")
        for i, kpt_file in enumerate(kpt_files):
            if len(kpt_files) > 1:
                kpt_filename = f"KPOINTS_band_split_{i + 1:0d}"
            else:
                kpt_filename = "KPOINTS"
            kpt_path = Path(directory) / kpt_filename if directory else Path(kpt_filename)
            kpt_file.write_file(kpt_path)

    return folders


def write_band_structure_calculation(
        structure: Structure,
        kpath: tuple[list[NDArray], list[str]],
        band_directory: str | Path,
        functional: str,
        splits: int,
        patches: list[str] | None = None,
        scf_charge: str | None = None,
        scf_kpoints: str | None = None,
        cartesian: bool = False,
        user_incar_settings: dict[str, Any] | None = None,
) -> None:
    """Generate and write a band structure calculation setup for VASP.

    Prepares KPOINTS paths, INCAR settings and structure files. Supports
    hybrid-functional and GGA calculations, split k-point paths, and copying
    a precomputed charge density for non-self-consistent runs.

    Parameters
    ----------
    structure : Structure
        Atomic structure used for the calculation.
    kpath : tuple of (list of numpy.ndarray, list of str)
        K-point coordinates along the band path and the high-symmetry
        labels, as returned by ``generate_band_structure_path``.
    band_directory : str or Path
        Output directory for the band structure inputs.
    functional : str
        Exchange-correlation functional, e.g. ``"PBE"`` or ``"HSE06"``.
    splits : int
        Number of segments to split the k-point path into separate runs.
    patches : list of str or None, optional
        Input patches applied to the VASP inputs. Default is None, treated
        as no patches.
    scf_charge : str or None, optional
        Path to a converged CHGCAR file; required for GGA functionals.
    scf_kpoints : str or None, optional
        Path to the SCF KPOINTS file; required for hybrid functionals.
    cartesian : bool, optional
        Whether k-points are given in Cartesian coordinates. Default is
        ``False``.
    user_incar_settings : dict or None, optional
        Additional INCAR settings provided by the user. Default is None.
    """
    hybrid = functional in ["PBE0", "HSE06", "DD_hybrid", "R2SCAN"]

    if hybrid:
        if scf_kpoints is None:
            print(
                "ERROR: SCF irreducible k-points are required for band structure calculations with a hybrid functional")
            return

        else:
            # ibz = _parse_ibzkpt(scf_kpoints)
            ibz = Kpoints.from_file(scf_kpoints)

    else:
        if scf_charge is None:
            print("ERROR: Converged charge density is required for band structure calculations with a GGA functional")
            return

        else:
            ibz = None

    kpoints, labels = kpath

    if splits > 1:
        make_folders = True
        kpts_per_split = math.ceil(len(kpoints) / splits)

    else:
        make_folders = False
        kpts_per_split = None

        Path(band_directory).mkdir(parents=True, exist_ok=True)

    folders = _write_kpoint_files(
        directory=band_directory,
        kpoints=kpoints,
        labels=labels,
        make_folders=make_folders,
        ibzkpt=ibz,
        kpts_per_split=kpts_per_split,
        cart_coords=cartesian,
    )

    incar_settings = {"KSPACING": 0}  # KSPACING is a dummy value to prevent overwriting of band path KPOINTS file(s)

    if user_incar_settings is not None:
        incar_settings = incar_settings | user_incar_settings

    if not hybrid:
        incar_settings["ICHARG"] = 11

    for folder in folders:
        directory = Path(band_directory) / folder

        write_vasp_calculation(
            structure=structure,
            recipe=functional,
            out_dir=directory,
            patches=patches,
            user_incar_settings=incar_settings)

        if not hybrid:
            shutil.copy(src=scf_charge, dst=directory / "CHGCAR")  # type: ignore


def write_castep_band_structure_calculation(
        cell_file: str | Path,
        kpath: tuple[list[NDArray], list[str]],
        band_directory: str | Path,
        splits: int = 1,
) -> None:
    """Generate and write a band structure calculation setup for CASTEP.

    Builds on a converged SCF calculation: the band path is appended to a
    copy of the SCF ``.cell``, the sibling ``.param`` is copied with the
    task switched to a spectral band structure, and the sibling ``.check``
    file - when present - is copied with the ``reuse`` tag set, which is
    CASTEP's equivalent of restarting from a converged charge density.

    Parameters
    ----------
    cell_file : str or Path
        Path to the SCF ``.cell`` file; its sibling ``.param`` is required
        and a sibling ``.check`` is picked up when present.
    kpath : tuple of (list of numpy.ndarray, list of str)
        K-point coordinates along the band path and the high-symmetry
        labels, as returned by ``generate_band_structure_path``.
    band_directory : str or Path
        Output directory for the band structure inputs. Must not already
        contain ``split-*`` folders.
    splits : int, optional
        Number of segments to split the k-point path into separate runs.
        Default is ``1``, which writes a single ``band.cell`` beside the
        adjusted ``.param`` and copied ``.check``.

    Raises
    ------
    FileNotFoundError
        If ``cell_file`` or its sibling ``.param`` does not exist.
    """
    cell_path = Path(cell_file)
    if not cell_path.is_file():
        raise FileNotFoundError(f"No CASTEP cell file at {cell_path}")

    param_path = cell_path.with_suffix(".param")
    if not param_path.is_file():
        raise FileNotFoundError(
            f"No .param file beside {cell_path}; the SCF calculation's .param"
            " is needed to derive the band structure task."
        )

    kpoints, labels = kpath

    band_path = Path(band_directory)
    band_path.mkdir(parents=True, exist_ok=True)

    if splits > 1:
        # sumo copies the .param (switching the task tags) and any .check
        # (setting reuse) into each split folder alongside the cell.
        castep_write_kpoint_files(
            str(cell_path),
            kpoints,
            labels,
            make_folders=True,
            kpts_per_split=math.ceil(len(kpoints) / splits),
            directory=str(band_path),
        )
        return

    castep_write_kpoint_files(
        str(cell_path),
        kpoints,
        labels,
        make_folders=False,
        directory=str(band_path),
    )

    # sumo's non-folder mode writes only band.cell, so complete the setup the
    # way its folder mode would: adjusted .param, plus the .check for reuse.
    tags = {"task": "Spectral", "spectral_task": "BandStructure"}
    check_path = cell_path.with_suffix(".check")
    if check_path.is_file():
        shutil.copy(check_path, band_path / check_path.name)
        tags["reuse"] = check_path.name
    copy_param(str(param_path), str(band_path), tags=tags)


def _is_soc_vasprun(vr: BSVasprun) -> bool:
    """Determine whether a VASP calculation includes spin-orbit coupling.

    Parameters
    ----------
    vr : BSVasprun
        Parsed VASP run object with INCAR settings and calculation metadata.

    Returns
    -------
    bool
        True if spin-orbit coupling was enabled (``LSORBIT = True``);
        False otherwise, or if the INCAR data cannot be accessed.
    """
    try:
        return bool(vr.incar.get("LSORBIT", False))
    except Exception:
        return False


def _castep_cell_for(bands_file: Path) -> str | None:
    """Find a sibling .cell file usable for high-symmetry labels, if any.

    Parameters
    ----------
    bands_file : Path
        Path of the ``.bands`` file being read.

    Returns
    -------
    str or None
        Path of the sibling ``.cell`` when it exists and contains a k-point
        path/list block; None otherwise. The block check matters: sumo's
        label reader exits the interpreter on a cell without one.
    """
    cell_path = bands_file.with_suffix(".cell")
    if not cell_path.is_file():
        return None
    if _CASTEP_KPOINT_BLOCK_RE.search(cell_path.read_text(encoding="utf-8")):
        return str(cell_path)
    return None


def get_band_structure(
        band_directory: str | Path, splits: int, code: str = "vasp"
) -> BandStructureSymmLine:
    """Load and reconstruct a symmetry-line band structure from calculation outputs.

    Reads one or more output files (including split band calculations) and
    reconstructs the full band structure along the high-symmetry k-path.

    Parameters
    ----------
    band_directory : str or Path
        Directory containing the output files: ``vasprun.xml`` for VASP, a
        ``<seed>.bands`` file for CASTEP (with high-symmetry labels read
        from a sibling ``.cell`` when present). Split calculations are
        expected in subfolders named ``"split-*"``.
    splits : int
        Number of split calculations used. If greater than 1, one output
        file per split directory is read; otherwise a single file.
    code : str, optional
        Which code produced the outputs, ``"vasp"`` or ``"castep"``.
        Default is ``"vasp"``.

    Returns
    -------
    BandStructureSymmLine
        Reconstructed band structure with eigenvalues along the full
        symmetry-line path.

    Raises
    ------
    ValueError
        If ``code`` is not ``"vasp"`` or ``"castep"``, or a single-file
        CASTEP directory holds several ``.bands`` files.
    FileNotFoundError
        If no CASTEP ``.bands`` file matches.
    """
    if code == "castep":
        if splits > 1:
            # Mirrors the VASP branch below - including reading every split
            # regardless of the argument's value; see the pinned
            # test_splits_argument_is_honoured xfail.
            bands_files = sorted(
                Path(band_directory).glob("split-*/*.bands"),
                key=lambda p: int(p.parent.name.split("-")[-1])
            )
            if not bands_files:
                raise FileNotFoundError(
                    f"No split-*/*.bands files found in {band_directory}"
                )
        else:
            bands_files = sorted(Path(band_directory).glob("*.bands"))
            if not bands_files:
                raise FileNotFoundError(f"No *.bands file found in {band_directory}")
            if len(bands_files) > 1:
                names = ", ".join(f.name for f in bands_files)
                raise ValueError(
                    f"Several .bands files in {band_directory}: {names};"
                    " point band_directory at a single calculation."
                )

        castep_bandstructures = [
            castep_band_structure(str(f), cell_file=_castep_cell_for(f))
            for f in bands_files
        ]
        reconstructed: BandStructureSymmLine = get_reconstructed_band_structure(
            castep_bandstructures
        )
        return reconstructed

    if code != "vasp":
        raise ValueError(f"Unsupported code {code!r}; expected 'vasp' or 'castep'.")

    if splits > 1:
        vaspruns = sorted(
            Path(band_directory).glob("split-*/vasprun.xml"),
            key=lambda p: int(p.parent.name.split("-")[-1])
        )

    else:
        vaspruns = [Path(band_directory) / "vasprun.xml"]

    bandstructures = []
    for vr_file in vaspruns:
        vr = BSVasprun(vr_file, parse_projected_eigen=False)
        bandstructures.append(vr.get_band_structure(line_mode=True, efermi="smart"))

    bs: BandStructureSymmLine = get_reconstructed_band_structure(bandstructures)

    return bs


# Simplified version of sumo.cli.bandplot
def plot_band_structure(
        bs: BandStructureSymmLine,
        plt: ModuleType,
        ymin: float = -6.0,
        ymax: float = 6.0,
        ylabel: str = "Energy (eV)",

        dos_file: str | Path | None = None,
        dos_label: str | None = None,
        total_only: bool = False,
        plot_total: bool = True,
        gaussian: float | None = None,
        yscale: float = 1,
        legend_cutoff: int = 3,

        vbm_cbm_marker: bool = False,
        projection_selection: list[Any] | None = None,
        mode: str = "rgb",
        normalise: str = "all",
        interpolate_factor: int = 4,
        color1: str = "#FF0000",
        color2: str = "#0000FF",
        color3: str = "#00FF00",
        colorspace: str = "lab",
        circle_size: float = 150,

        scissor: float | None = None,
        zero_line: bool = False,
        zero_energy: float | None = None,

        elements: list[Any] | None = None,
        lm_orbitals: list[Any] | None = None,
        atoms: list[Any] | None = None,
        spin: bool | None = None,

        colours: dict[str, Any] | None = None,

        style: str | None = None,
        no_base_style: bool = False,

        code: str = "vasp",

) -> ModuleType:
    """Plot a band structure, optionally with a projected view and density of states.

    A simplified interface over sumo's plotters, with scissor correction,
    VBM/CBM markers and energy alignment.

    Parameters
    ----------
    bs : BandStructureSymmLine
        Band structure object with k-point paths and eigenvalues.
    plt : module
        Matplotlib or compatible plotting interface used for rendering.
    ymin : float, optional
        Minimum energy in eV on the y-axis. Default is ``-6.0``.
    ymax : float, optional
        Maximum energy in eV on the y-axis. Default is ``6.0``.
    ylabel : str, optional
        Label for the energy axis. Default is ``"Energy (eV)"``.
    dos_file : str or Path or None, optional
        Path to density of states data: ``vasprun.xml`` for VASP, a
        ``<seed>.bands`` file for CASTEP (sibling ``.pdos_bin`` and
        ``.cell`` files are picked up automatically). When provided, the
        DOS is plotted alongside the band structure. Default is None.
    dos_label : str or None, optional
        Label for the DOS plot. Default is None.
    total_only : bool, optional
        Plot only the total DOS. Default is ``False``.
    plot_total : bool, optional
        Include the total DOS in the plot. Default is ``True``.
    gaussian : float or None, optional
        Gaussian smearing applied to the DOS. Default is None.
    yscale : float, optional
        Scaling factor for the DOS axis. Default is ``1``.
    legend_cutoff : int, optional
        Threshold for legend simplification. Default is ``3``.
    vbm_cbm_marker : bool, optional
        Mark the valence band maximum and conduction band minimum.
        Default is ``False``.
    projection_selection : list or None, optional
        Orbital/element projections for projected band structure plotting.
        Default is None.
    mode : str, optional
        Projection visualisation mode. Default is ``"rgb"``.
    normalise : str, optional
        Normalisation method for projections. Default is ``"all"``.
    interpolate_factor : int, optional
        Interpolation factor for smoothing bands. Default is ``4``.
    color1 : str, optional
        First colour for projected band visualisation. Default is
        ``"#FF0000"``.
    color2 : str, optional
        Second colour for projected band visualisation. Default is
        ``"#0000FF"``.
    color3 : str, optional
        Third colour for projected band visualisation. Default is
        ``"#00FF00"``.
    colorspace : str, optional
        Colour mapping space. Default is ``"lab"``.
    circle_size : float, optional
        Size of the projection markers. Default is ``150``.
    scissor : float or None, optional
        Band gap correction applied to the band structure. Default is None.
    zero_line : bool, optional
        Draw a horizontal reference line at zero energy. Default is
        ``False``.
    zero_energy : float or None, optional
        Reference energy level for alignment. Default is None.
    elements : list or None, optional
        Element selection for the DOS projection. Default is None.
    lm_orbitals : list or None, optional
        Orbital selection for the DOS projection. Default is None.
    atoms : list or None, optional
        Atomic selection for the DOS projection. Default is None.
    spin : bool or None, optional
        Spin-polarised plotting option. Default is None.
    colours : dict or None, optional
        Custom colour mapping for the DOS and bands. Default is None.
    style : str or None, optional
        Plotting style preset. Default is None.
    no_base_style : bool, optional
        Disable the default plotting style. Default is ``False``.
    code : str, optional
        Which code produced the ``dos_file``, ``"vasp"`` or ``"castep"``.
        Only consulted when a DOS panel is drawn. Default is ``"vasp"``.

    Returns
    -------
    module
        The plotting module with the rendered band structure, and the DOS
        when included.

    Raises
    ------
    ValueError
        If ``code`` is not ``"vasp"`` or ``"castep"``, or a scissor
        correction is requested for a CASTEP DOS, which sumo's reader does
        not support.
    """
    if code not in ("vasp", "castep"):
        raise ValueError(f"Unsupported code {code!r}; expected 'vasp' or 'castep'.")

    if projection_selection and mode == "rgb" and len(projection_selection) > 3:
        print(
            "ERROR: RGB projected band structure only "
            "supports up to 3 elements/orbitals."
            "\nUse alternative --mode setting."
        )
        return plt

    dos_plotter = None
    dos_opts = None
    if dos_file:
        if code == "castep":
            if scissor:
                raise ValueError(
                    "A scissor correction is not supported with CASTEP DOS data."
                )
            bands_path = Path(dos_file)
            pdos_file = bands_path.with_suffix(".pdos_bin")
            cell_file = bands_path.with_suffix(".cell")
            # A projected DOS needs both siblings; with either missing, fall
            # back to the total DOS the way sumo's own CLI does.
            has_pdos = pdos_file.is_file() and cell_file.is_file()
            dos, pdos = castep_read_dos(
                str(bands_path),
                pdos_file=str(pdos_file) if has_pdos else None,
                cell_file=str(cell_file) if cell_file.is_file() else None,
                gaussian=gaussian,
                lm_orbitals=lm_orbitals,
                elements=elements,
                atoms=atoms,
                total_only=total_only,
            )
        else:
            dos, pdos = load_dos(
                dos_file,
                elements,
                lm_orbitals,
                atoms,
                gaussian,
                total_only,
                scissor=scissor,
            )

        dos_plotter = SDOSPlotter(dos, pdos)
        dos_opts = {
            "plot_total": plot_total,
            "legend_cutoff": legend_cutoff,
            "colours": colours,
            "yscale": yscale,
        }

    if scissor:
        bs = bs.apply_scissor(scissor)

    plotter = SBSPlotter(bs)
    if projection_selection:
        plt = plotter.get_projected_plot(
            projection_selection,
            mode=mode,
            normalise=normalise,
            interpolate_factor=interpolate_factor,
            color1=color1,
            color2=color2,
            color3=color3,
            colorspace=colorspace,
            circle_size=circle_size,
            zero_to_efermi=True,
            zero_line=zero_line,
            zero_energy=zero_energy,
            ymin=ymin,
            ymax=ymax,
            height=None,
            width=None,
            vbm_cbm_marker=vbm_cbm_marker,
            ylabel=ylabel,
            plt=plt,
            dos_plotter=dos_plotter,
            dos_options=dos_opts,
            dos_label=dos_label,
            fonts=None,
            style=style,
            no_base_style=no_base_style,
            spin=spin,
            title=None,
        )

    else:
        plt = plotter.get_plot(
            zero_to_efermi=True,
            zero_line=zero_line,
            zero_energy=zero_energy,
            ymin=ymin,
            ymax=ymax,
            height=None,
            width=None,
            vbm_cbm_marker=vbm_cbm_marker,
            ylabel=ylabel,
            plt=plt,
            dos_plotter=dos_plotter,
            dos_options=dos_opts,
            dos_label=dos_label,
            fonts=None,
            style=style,
            no_base_style=no_base_style,
            spin=spin,
            title=None,
        )

    return plt
