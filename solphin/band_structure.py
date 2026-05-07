from typing import Optional, Any
from pathlib import Path
from glob import glob
import math
import numpy as np
from numpy.typing import NDArray
import os
import shutil

from pymatgen.core.structure import Structure
from pymatgen.io.vasp.outputs import BSVasprun
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.io.vasp.inputs import Kpoints

from sumo.symmetry.kpoints import get_path_data

# from sumo.cli.kgen import _parse_ibzkpt
from sumo.plotting.bs_plotter import SBSPlotter
from sumo.plotting.dos_plotter import SDOSPlotter
from sumo.electronic_structure.dos import load_dos
from sumo.electronic_structure.bandstructure import get_reconstructed_band_structure

from solphin.vasp_inputs import write_vasp_calculation

import logging
logger = logging.getLogger()
logger.setLevel(logging.WARNING)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def generate_band_structure_path(
        structure:Structure,
        definition:str="bradcrack",
        symprec:float=0.01,
        density:int=60,
        cartesian:bool=False
                            ) -> tuple[Structure, tuple[NDArray, list[str]]]:
    
    """
    Generates a high-symmetry k-point path for band structure calculations.

    This function constructs a Brillouin-zone path based on a specified symmetry
    definition, ensuring consistency with a canonical primitive structure if needed.
    It supports different k-path conventions and optionally converts coordinates
    into Cartesian space.

    If the input structure differs from the canonical primitive cell, the path is
    recomputed using the primitive structure for consistency.

    Parameters:
        structure(Structure): input crystal structure.
        definition(string): k-path generation scheme (e.g. "bradcrack").
            Default is "bradcrack".
        symprec(float): symmetry tolerance used for structure analysis.
            Default is 0.01.
        density(int): number of k-points per unit length along the path.
            Higher values produce smoother band structures.
            Default is 60.
        cartesian(bool): if True, returns k-points in Cartesian coordinates;
            otherwise returns reciprocal coordinates.

    Returns:
        tuple:
            canonical_structure(Structure): primitive/canonical structure used
                for k-path generation.
            (canonical_kpoints(np.array), canonical_labels(list[str])):
                tuple containing the k-point path and corresponding labels.
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

    return canonical_structure, (canonical_kpoints, canonical_labels) # type: ignore

#Simplified version of sumo.io.vasp.write_kpoint_files
def write_kpoint_files(
    directory:str|Path,
    kpoints:NDArray,
    labels:list[str],
    make_folders:bool=True,
    ibzkpt:Optional[Kpoints]=None,
    kpts_per_split:Optional[int]=None,
    cart_coords:bool=False,
) -> list[str]:
    
    """
    Generates and writes KPOINTS files for band structure calculations, optionally split into multiple segments.

    This function takes a band path defined by k-points and labels, optionally splits it into
    multiple segments for parallel or chunked calculations, and writes VASP-compatible KPOINTS
    files either into separate folders or as individual files.

    It also supports hybrid calculations by incorporating irreducible k-point meshes (IBZ)
    and adjusting k-point weights accordingly.

    Parameters:
        directory(string or Path): output directory where KPOINTS files or folders are written.
        kpoints(NDArray): array of k-point coordinates defining the band path.
        labels(list[str]): high-symmetry point labels corresponding to kpoints.
        make_folders(bool): if True, creates separate folders for each split segment.
            Default is True.
        ibzkpt(Kpoints or None): irreducible Brillouin zone k-point mesh for hybrid calculations.
            If provided, k-point weights are set accordingly.
        kpts_per_split(int or None): number of k-points per split segment.
            If None, no splitting is performed.
        cart_coords(bool): if True, treats kpoints as Cartesian coordinates;
            otherwise uses reciprocal coordinates.

    Returns:
        list[str]: list of folder names (or empty strings if folders are not used)
            corresponding to each generated KPOINTS segment.
    """

    if kpts_per_split:
        kpt_splits = [
            kpoints[i : i + kpts_per_split]
            for i in range(0, len(kpoints), kpts_per_split)
        ]
        label_splits = [
            labels[i : i + kpts_per_split]
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
            label_split = [""] * len(ibzkpt.labels) + label_split
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

            if directory:
                folder = os.path.join(directory, folder)

            try:
                os.makedirs(folder)
            except OSError:
                # print(f"Directory {folder} already exists.")
                pass

            kpt_file.write_file(os.path.join(folder, "KPOINTS"))

    else:
        folders.append("")
        for i, kpt_file in enumerate(kpt_files):
            if len(kpt_files) > 1:
                kpt_filename = f"KPOINTS_band_split_{i + 1:0d}"
            else:
                kpt_filename = "KPOINTS"
            if directory:
                kpt_filename = os.path.join(directory, kpt_filename)
            kpt_file.write_file(kpt_filename)

    return folders

def write_band_structure_calculation(
        structure:Structure,
        kpath:tuple[NDArray, list[str]],
        band_directory:str|Path,
        functional:str,
        splits:int,
        patches:list[str]=[],
        scf_charge:Optional[str]=None,
        scf_kpoints:Optional[str]=None,
        cartesian:bool=False,
        user_incar_settings:Optional[dict[str,Any]]=None):
    """
    Generates and writes a band structure calculation setup for VASP.

    This function prepares input files for band structure calculations, including
    KPOINTS paths, INCAR settings, and structure files. It supports both hybrid
    functional and GGA calculations, handles split k-point path calculations, and
    optionally copies precomputed charge densities for non-self-consistent runs.

    Parameters:
        structure(Structure): atomic structure used for the calculation.
        kpath(tuple[NDArray, list[str]]): tuple containing:
            - kpoints (NDArray): array of k-point coordinates along the band path
            - labels (list[str]): high-symmetry point labels for plotting
        band_directory(string or Path): output directory for band structure inputs.
        functional(string): exchange-correlation functional (e.g. PBE, HSE06).
        splits(int): number of segments to split the k-point path into separate runs.
        patches(list[str]): optional list of input patches applied to VASP inputs.
            Default is empty list.
        scf_charge(string or None): path to converged CHGCAR file (required for GGA).
        scf_kpoints(string or None): path to SCF KPOINTS file (required for hybrid functionals).
        cartesian(bool): whether k-points are given in Cartesian coordinates.
        user_incar_settings(dict or None): additional INCAR settings provided by user.

    Returns:
        None
    """

    hybrid = functional in ["PBE0", "HSE06", "DD_hybrid", "R2SCAN"]
    
    if hybrid:
        if scf_kpoints is None:
            print("ERROR: SCF irreducible k-points are required for band structure calculations with a hybrid functional")
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
        kpts_per_split = math.ceil(len(kpoints)/splits)
    
    else:
        make_folders = False
        kpts_per_split = None

        try:
            os.mkdir(f"{band_directory}")
        except OSError:
            pass

    folders = write_kpoint_files(
        directory=band_directory,
        kpoints=kpoints,
        labels=labels,
        make_folders=make_folders,
        ibzkpt=ibz,
        kpts_per_split=kpts_per_split,
        cart_coords=cartesian,
    )

    incar_settings = {"KSPACING":0} #KSPACING is a dummy value to prevent overwriting of band path KPOINTS file(s)

    if user_incar_settings is not None:
        incar_settings = incar_settings | user_incar_settings

    if not hybrid:
        incar_settings["ICHARG"] = 11

    for folder in folders:
        directory = f"{band_directory}/{folder}"

        write_vasp_calculation(
            structure=structure,
            recipe=functional,
            out_dir=directory,
            patches = patches,
            user_incar_settings=incar_settings)
        
        if not hybrid:
            shutil.copy(src=scf_charge, dst=os.path.join(directory, "CHGCAR")) # type: ignore


def get_band_structure(band_directory:str|Path, splits:int) -> BandStructureSymmLine:
    """
    Loads and reconstructs a symmetry-line band structure from VASP calculation outputs.

    This function reads one or multiple vasprun.xml files (including split band
    calculations), extracts band structures using pymatgen's BSVasprun parser,
    and reconstructs a full band structure along a high-symmetry k-path.

    Parameters:
        band_directory(string or Path): directory containing vasprun.xml files.
            If multiple split calculations are used, they are expected in subfolders
            named "split-*".
        splits(int): number of split calculations used.
            If greater than 1, the function searches for multiple vasprun.xml files
            in split directories; otherwise it reads a single file.

    Returns:
        BandStructureSymmLine: reconstructed band structure object containing
            eigenvalues along the full symmetry line path.
    """

    if splits > 1:
        vaspruns = glob(f"{band_directory}/split-*/vasprun.xml")

    else:
        vaspruns = [f"{band_directory}/vasprun.xml"]

    bandstructures = []
    for vr_file in vaspruns:
        vr = BSVasprun(vr_file, parse_projected_eigen=False)
        bs = vr.get_band_structure(line_mode=True, efermi="smart")
        bandstructures.append(bs)
        
    bs:BandStructureSymmLine = get_reconstructed_band_structure(bandstructures)

    return bs

#Simplified version of sumo.cli.bandplot
def plot_band_structure(
    bs:BandStructureSymmLine, 
    plt,
    ymin=-6.0,
    ymax=6.0,
    ylabel="Energy (eV)",

    dos_file=None,
    dos_label=None,
    total_only=False,
    plot_total=True,
    gaussian=None,
    yscale=1,
    legend_cutoff=3,

    vbm_cbm_marker=False,
    projection_selection=None,
    mode="rgb",
    normalise="all",
    interpolate_factor=4,
    color1="#FF0000",
    color2="#0000FF",
    color3="#00FF00",
    colorspace="lab",
    circle_size=150,

    scissor=None,
    zero_line=False,
    zero_energy=None,

    elements=None,
    lm_orbitals=None,
    atoms=None,
    spin=None,

    colours=None,

    style=None,
    no_base_style=False,

):
    """
    Plots a band structure (and optionally density of states) for a symmetry-line band structure object.

    This function provides a simplified interface for plotting electronic band structures,
    optionally including projected band structures and density of states (DOS), with
    customizable visual styling and analysis features such as scissor correction,
    VBM/CBM markers, and energy alignment.

    Parameters:
        bs(BandStructureSymmLine): band structure object containing k-point paths
            and eigenvalues.
        plt: Matplotlib or compatible plotting interface used for rendering.
        ymin(float): minimum energy value (eV) for plot y-axis. Default is -6.0.
        ymax(float): maximum energy value (eV) for plot y-axis. Default is 6.0.
        ylabel(string): label for the energy axis. Default is "Energy (eV)".

        dos_file(string or None): file path to density of states data. If provided,
            DOS is plotted alongside the band structure.
        dos_label(string or None): label for DOS plot.
        total_only(bool): whether to plot only total DOS.
        plot_total(bool): whether to include total DOS in plot.
        gaussian(float or None): Gaussian smearing applied to DOS.
        yscale(float): scaling factor for DOS axis.
        legend_cutoff(int): threshold for legend simplification.

        vbm_cbm_marker(bool): whether to mark valence band maximum and conduction
            band minimum.
        projection_selection(list or None): orbital/element projections for
            projected band structure plotting.
        mode(string): projection visualization mode (e.g. "rgb").
        normalise(string): normalization method for projections.
        interpolate_factor(int): interpolation factor for smoothing bands.
        color1, color2, color3(string): colors used for projected band visualization.
        colorspace(string): color mapping space (e.g. "lab").
        circle_size(float): size of projection markers.

        scissor(float or None): band gap correction applied to band structure.
        zero_line(bool): whether to draw a horizontal reference line at zero energy.
        zero_energy(float or None): reference energy level for alignment.

        elements(list or None): element selection for DOS projection.
        lm_orbitals(list or None): orbital selection for DOS projection.
        atoms(list or None): atomic selection for DOS projection.
        spin(bool or None): spin-polarized plotting option.

        colours(dict or None): custom color mapping for DOS/bands.
        style(string or None): plotting style preset.
        no_base_style(bool): disables default plotting style.

    Returns:
        plt: modified plotting object with the rendered band structure (and DOS if included).
    """

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
