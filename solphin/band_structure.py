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
    Generate the high-symmetry k-point path for a band structure calculation. Adapted from sumo.cli.kgen.get_kpoint_path_data.

    :param structure: The input structure for which to generate the k-point path.
    :type structure: :obj:`~pymatgen.core.structure.Structure`
    :param definition: The definition of the high-symmetry path. Options are:
        "bradcrack": The Bradley and Cracknell standard path.
        "pymatgen": The path defined by pymatgen. 
        "latimer-munro": The path defined by Latimer and Munro.
        "seekpath": The path defined by SeeK-path.
    :type definition: str, optional
    :param symprec: The symmetry precision for determining the space group. Default is 0.01.
    :type symprec: float, optional
    :param density: The density of k-points along the path. Default is 60.
    :type density: int, optional
    :param cartesian: Whether to return the k-points in cartesian coordinates. Default is False (fractional coordinates).
    :type cartesian: bool, optional
    :return: The canonical structure and a tuple of k-points and labels.
    :rtype: tuple[Structure, tuple[NDArray, list[str]]]
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
    Write KPOINTS files for band structure calculations. Adapted from sumo.io.vasp.write_kpoint_files.

    :param directory: The directory in which to write the KPOINTS files.
    :type directory: str | Path
    :param kpoints: The k-points along the high-symmetry path. Should be provided as a :obj:`numpy.ndarray` of shape (N, 3), where N is the number of k-points.
    :type kpoints: :obj:`numpy.ndarray`
    :param labels: The k-point labels. Should be provided as a :obj:`list` of strings, with the same length as the number of k-points. Each label should correspond to the k-point at the same index in the ``kpoints`` array. If a k-point has no label, the corresponding entry in the ``labels`` list should be an empty string.
    :type labels: list[str]
    :param make_folders: Whether to create separate folders for each segment of the band structure calculation. Default is True, which will create folders named "split-01", "split-02", etc. If False, all KPOINTS files will be written to the specified directory with names "KPOINTS_band_split_1", "KPOINTS_band_split_2", etc. if there are multiple splits, or "KPOINTS" if there is only one split.
    :type make_folders: bool
    :param ibzkpt: An optional Kpoints object containing the irreducible k-points from a previous SCF calculation. If provided, these k-points will be included in the KPOINTS files with weights set to 0, and the band structure k-points will be added on top with weights set to 1. This is necessary for hybrid functional calculations.
    :type ibzkpt: Kpoints, optional
    :param kpts_per_split: The number of k-points to include in each split. If not set, all k-points will be included in a single split. This is useful for breaking up long band paths into multiple calculations.
    :type kpts_per_split: int, optional
    :param cart_coords: Whether to return the k-points in cartesian coordinates. Default is False (fractional coordinates).
    :type cart_coords: bool
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
    Write the input files for a band structure calculation.

    :param structure: The input structure for which to generate the band structure calculation.
    :type structure: :obj:`~pymatgen.core.structure.Structure`
    :param kpath: A tuple of the k-points along the high-symmetry path and the corresponding labels. The k-points should be provided as a :obj:`numpy.ndarray` of shape (N, 3), where N is the number of k-points, and the labels should be provided as a :obj:`list` of strings, with the same length as the number of k-points. Each label should correspond to the k-point at the same index in the k-points array. If a k-point has no label, the corresponding entry in the labels list should be an empty string.
    :type kpath: tuple[NDArray, list[str]]
    :param band_directory: The directory in which to write the input files for the band structure calculation.
    :type band_directory: str | Path
    :param functional: The exchange-correlation functional to use for the band structure calculation. Options are "PBE", "PBE0", "HSE06", "DD_hybrid", and "R2SCAN".
    :type functional: str
    :param splits: The number of splits to break the band structure calculation into. If greater than 1, the k-points will be divided into approximately equal segments and separate calculations will be written for each segment. This is useful for long band paths that may be too computationally expensive to calculate in a single calculation. Default is 1 (no splits).
    :type splits: int
    :param patches: A list of patches to apply to the INCAR settings.
    :type patches: list[str]
    :param scf_charge: The path to the CHGCAR file from a previous SCF calculation. This is required for band structure calculations with a GGA functional, but not for hybrid functional calculations.
    :type scf_charge: str, optional
    :param scf_kpoints: The path to the KPOINTS file containing the irreducible k-points from a previous SCF calculation. This is required for band structure calculations with a hybrid functional, but not for GGA functional calculations.
    :type scf_kpoints: str, optional
    :param cartesian: Whether to return the k-points in cartesian coordinates. Default is False (fractional coordinates).
    :type cartesian: bool
    :param user_incar_settings: A dictionary of additional INCAR settings to include in the band structure calculation. These settings will be combined with the default settings for the chosen functional, and any settings specified in the ``patches`` list. If a setting is included in both ``user_incar_settings`` and ``patches``, the value from ``user_incar_settings`` will take precedence.
    :type user_incar_settings: dict[str, Any], optional
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
    Read the band structure from the output of a band structure calculation. Adapted from sumo.cli.bandplot.get_band_structure.

    :param band_directory: The directory containing the output files from the band structure calculation.
    :type band_directory: str | Path
    :param splits: The number of splits that the band structure calculation was broken into. If greater than 1, the method will look for vasprun.xml files in subdirectories named "split-01", "split-02", etc. If 1, the method will look for a single vasprun.xml file in the specified directory.
    :type splits: int
    :return: The band structure object containing the band structure data.
    :rtype: BandStructureSymmLine
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
    Plot the band structure. Adapted from sumo.cli.bandplot.plot_band_structure.
    
    :param bs: The band structure object containing the band structure data.
    :type bs: BandStructureSymmLine
    :param plt: The matplotlib.pyplot instance to use for plotting.
    :type plt: matplotlib.pyplot
    :param ymin: The minimum energy to display on the plot. Default is -6.0 eV.
    :type ymin: float, optional
    :param ymax: The maximum energy to display on the plot. Default is 6.0 eV.
    :type ymax: float, optional
    :param ylabel: The label for the y-axis. Default is "Energy (eV)".
    :type ylabel: str, optional
    :param dos_file: The path to the DOS file to plot alongside the band structure. If not set, no DOS will be plotted. Default is None.
    :type dos_file: str, optional
    :param dos_label: The label for the DOS plot. Default is None.
    :type dos_label: str, optional
    :param total_only: Whether to plot only the total DOS, or also the projected DOS. Default is False (plot both total and projected DOS).
    :type total_only: bool, optional
    :param plot_total: Whether to plot the total DOS. Default is True.
    :type plot_total: bool, optional
    :param gaussian: The width of the Gaussian smearing to apply to the DOS. Default is None (no smearing).
    :type gaussian: float, optional
    :param yscale: The scaling factor to apply to the DOS. Default is 1 (no scaling).
    :type yscale: float, optional
    :param legend_cutoff: The minimum contribution (in %) for a projected DOS component to be included in the legend. Default is 3%.
    :type legend_cutoff: float, optional
    :param vbm_cbm_marker: Whether to mark the VBM and CBM on the plot. Default is False.
    :type vbm_cbm_marker: bool, optional
    :param projection_selection: A list of the projections to include in the plot. Each projection should be specified as a string in the format "Element-Orbital", where "Element" is the chemical symbol of the element, and "Orbital" is the orbital type (e.g. "s", "p", "d"). For example, ["Fe-d", "O-p"].
    :type projection_selection: list[str], optional
    :param mode: The mode to use for the projected band structure plot. Options are "rgb" (the default), which will plot the projections as RGB colors on the band structure, and "markers", which will plot the projections as colored markers on the band structure. Note that "rgb" mode only supports up to 3 projections, while "markers" mode supports any number of projections. Default is "rgb".
    :type mode: str, optional
    :param normalise: The method to use for normalising the projection contributions. Options are "all" (the default), which will normalise the contributions for each k-point by the total contribution of the selected projections at that k-point, and "max", which will normalise the contributions for each projection by the maximum contribution of that projection across all k-points. Default is "all".
    :type normalise: str, optional
    :param interpolate_factor: The factor by which to interpolate the band structure for smoother plotting. Default is 4 (i.e. the band structure will be interpolated to have 4 times as many k-points along the path).
    :type interpolate_factor: int, optional
    :param color1: The color to use for the first projection in "rgb" mode. Default is "#FF0000" (red).
    :type color1: str, optional
    :param color2: The color to use for the second projection in "rgb" mode. Default is "#0000FF" (blue).
    :type color2: str, optional
    :param color3: The color to use for the third projection in "rgb" mode. Default is "#00FF00" (green).
    :type color3: str, optional
    :param colorspace: The colorspace to use for the RGB projections. Options are "lab" (the default), which will use the CIELAB colorspace, and "hsv", which will use the HSV colorspace. Default is "lab".
    :type colorspace: str, optional
    :param circle_size: The size of the circles to use for the markers in "markers" mode. Default is 150.
    :type circle_size: int, optional
    :param scissor: The scissor correction to apply to the band structure. This should be provided as a float representing the energy (in eV) by which to shift the conduction band minimum. Default is None (no scissor correction).
    :type scissor: float, optional
    :param zero_line: Whether to include a horizontal line at the zero energy level. Default is False.
    :type zero_line: bool, optional
    :param zero_energy: The energy (in eV) to set as the zero energy level. If not set, the Fermi energy from the band structure will be used as the zero energy level. Default is None.
    :type zero_energy: float, optional
    :param elements: A list of the elements to include in the projected band structure plot. If not set, all elements will be included. Default is None.
    :type elements: list[str], optional
    :param lm_orbitals: A list of the orbitals to include in the projected band structure plot. Each orbital should be specified as a string in the format "lm", where "l" is the orbital angular momentum quantum number (0 for s, 1 for p, 2 for d, etc.) and "m" is the magnetic quantum number (e.g. 0 for s, -1, 0, 1 for p, etc.). For example, ["0", "1", "2"] would include s, p, and d orbitals. If not set, all orbitals will be included. Default is None.
    :type lm_orbitals: list[str], optional
    :param atoms: A list of the atom indices to include in the projected band structure plot. The atom indices should correspond to the indices of the atoms in the structure used to generate the band structure, and should be provided as a list of integers. For example, [0, 1, 2] would include the first three atoms in the structure. If not set, all atoms will be included. Default is None.
    :type atoms: list[int], optional
    :param spin: The spin channel to plot for a spin-polarised band structure. This should be provided as an integer, where 0 corresponds to the total band structure (for non-spin-polarised calculations), 1 corresponds to the spin-up channel, and 2 corresponds to the spin-down channel. Default is None, which will plot the total band structure for non-spin-polarised calculations, and the spin-up channel for spin-polarised calculations.
    :type spin: int, optional
    :param colours: A dictionary mapping the projection labels to the colors to use for those projections in the plot. The keys of the dictionary should correspond to the labels of the projections as they appear in the legend (e.g. "Fe-d", "O-p"), and the values should be the colors to use for those projections, specified as strings in a format recognized by matplotlib (e.g. "#FF0000" for red). If not set, default colors will be used for the projections. Default is None.
    :type colours: dict[str, str], optional
    :param style: The matplotlib style to use for the plot. This should be provided as a string corresponding to a valid matplotlib style (e.g. "ggplot", "seaborn", "bmh", etc.). If not set, the default matplotlib style will be used. Default is None.
    :type style: str, optional
    :param no_base_style: Whether to use the default sumo style. If True, the default sumo style will not be applied to the plot, and only the style specified in the ``style`` parameter (if any) will be used.
    :type no_base_style: bool, optional
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
