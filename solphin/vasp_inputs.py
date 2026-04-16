from typing import Dict, List, Optional, Union
from importlib.resources import files
import json 
import solphin.resources

from monty.serialization import loadfn
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.sets import VaspInputSet
from pymatgen.io.vasp import Kpoints

def read_structure_pmg(filename):
    """
    Reads a structure file (POSCAR/CONTCAR, CIF, XYZ, etc.) into a pymatgen Structure object.
    
    Parameters
    ----------
    filename : str
        Path to the structure file. Supported formats include:
        - VASP POSCAR/CONTCAR
        - CIF (.cif)
        - XYZ (.xyz)
        - JSON (.json)
        - Other pymatgen-supported formats
    
    Returns
    -------
    Structure
        A pymatgen Structure object with lattice, species, and atomic positions.
    """
    structure = Structure.from_file(filename)
    return structure

def _load_config(fname: str) -> Dict[str, Dict[str, Union[str, Dict[str, str]]]]:
    """
    Load configuration information from a JSON file.

    Args:
        fname (str): The name of the JSON file to load.

    Returns:
        dict: A dictionary containing the configuration information.
    """
    resource_path = files("solphin.resources") / fname  # adjust module name
    with resource_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    return config

def determine_potcar_functional(
    recipe: str,
    potcar_functional: Optional[str],
    config: Dict[str, Dict[str, Union[str, Dict[str, str]]]],
) -> str:
    """
    Determine the POTCAR functional to use based on the recipe.

    Args:
        recipe (str): The name of the recipe.
        potcar_functional (str, optional): The provided POTCAR functional.
        config (dict): The configuration dictionary.

    Returns:
        str: The determined POTCAR functional.
    """
    if potcar_functional:
        return potcar_functional
    if recipe == "LDA":
        return "LDA_64"
    return config["POTCAR_FUNCTIONAL"]


def prepare_incar(
    recipe: str,
    patches: List[str],
    config: Dict[str, Dict[str, Union[str, Dict[str, str]]]],
) -> Dict[str, Union[str, int, float]]:
    """
    Prepare the INCAR settings for the VASP calculation.

    Args:
        recipe (str): The name of the recipe.
        patches (list): A list of patches to apply.
        config (dict): The configuration dictionary.

    Returns:
        dict: The prepared INCAR settings.
    """

    incar = config["INCAR"][recipe]
    if recipe in ["HSE06", "PBE0"]:
        incar.update({"NCORE": 4})

    for patch in patches:
        if patch != "gamma_only":
            if patch == "defect":
                incar.update(config["PATCHES"]["relax_atoms"])
                incar.update(config["PATCHES"]["spin_polarised"])
            else:
                incar.update(config["PATCHES"][patch])
    return incar

def create_vasp_set(
    structure: Structure,
    incar: Dict[str, Union[str, int, float]],
    potcar_functional: str,
    config: Dict[str, Dict[str, Union[str, Dict[str, str]]]],
    **calc_kwargs,
) -> VaspInputSet:
    """
    Create the VASP set for the calculation.

    Args:
        structure (Structure): The structure to calculate.
        incar (dict): The INCAR settings.
        potcar_functional (str): The POTCAR functional to use.
        config (dict): The configuration dictionary.

    Returns:
        DictSet: The created VASP set.
    """
    return VaspInputSet(
        structure,
        {
            "INCAR": incar,
            "POTCAR": config["POTCAR"],
            "POTCAR_FUNCTIONAL": potcar_functional,
        },
        **calc_kwargs,
    )

def prepare_vdw_tags(recipe: str, patches: List[str]) -> Dict[str, Union[int, float]]:
    """
    Prepare the VDW tags for the INCAR settings.

    Args:
        recipe (str): The name of the recipe.
        patches (list): A list of patches to apply.

    Returns:
        dict: The prepared VDW tags.
    """
    if "vdw_d3_bj" in patches:
        if recipe == "HSE06":
            return {
                "IVDW": 12,
                "VDW_S8": 2.310,
                "VDW_A1": 0.383,
                "VDW_A2": 5.685,
            }
        return {"IVDW": 12}
    if "vdw_d3" in patches:
        if recipe == "HSE06":
            return {"IVDW": 11, "VDW_S8": 0.109, "VDW_SR": 1.129}
        return {"IVDW": 11}
    if "vdw_d4" in patches:
        if recipe == "HSE06":
            return {"IVDW": 13, 
                    "VDW_S8" : 1.19528249, 
                    "VDW_A1" : 0.38663183, 
                    "VDW_A2" : 5.19133469}
        return {"IVDW": 13}
    if "rvv10" in patches:
        if recipe == "R2SCAN":
            return {"LUSE_VDW" : True,
                    "BPARAM": 11.95,
                    "CPARAM": 0.0093}
    return {}

def apply_patches(
    vasp_set: VaspInputSet,
    patches: List[str],
    recipe: str,
    incar: Dict[str, Union[str, int, float]],
) -> None:
    """
    Apply patches to the VASP set.

    Args:
        vasp_set (DictSet): The VASP set.
        patches (list): A list of patches to apply.
        recipe (str): The name of the recipe.
        incar (dict): The INCAR settings.
    """
    if "relax_cell" in patches:
        encut = vasp_set.user_incar_settings.get("ENCUT", incar["ENCUT"]) * 1.3
        vasp_set.user_incar_settings["ENCUT"] = encut

    if "gamma_only" in patches:
        vasp_set.user_kpoints_settings = Kpoints(kpts=((1, 1, 1)))

    if "vdw_d3_bj" in patches or "vdw_d3" in patches or "vdw_d4":
        vdw_tags = prepare_vdw_tags(recipe, patches)
        vasp_set.user_incar_settings.update(vdw_tags)

    if "lobster" in patches and "NBANDS" not in vasp_set.user_incar_settings:
        vasp_set.user_incar_settings["NBANDS"] = vasp_set.estimate_nbands() * 2


def write_vasp_calculation(
    structure: Structure,
    recipe: str,
    out_dir: str,
    patches: Optional[List[str]] = None,
    potcar_functional: Optional[str] = None,
    **calc_kwargs,
) -> Optional[Dict[str, int]]:
    """
    Write a VASP calculation from a recipe.

    Args:
        structure (pymatgen.core.structure.Structure): The structure to calculate.
        recipe (str): The name of the recipe to use.
        out_dir (str): The output directory for the calculation.
        patches (list, optional): A list of patches to apply to the recipe. Defaults to None.
        get_stats (bool, optional): Whether to return statistics on the calculation. Defaults to False.
        potcar_functional (str, optional): The POTCAR functional to use. Defaults to None.

    Returns:
        dict: A dictionary containing the estimated number of bands and the number of electrons in the system, if get_stats is True.
    """
    patches = patches or []

    config = _load_config("base_recipes.json")
    potcar_functional = determine_potcar_functional(recipe, potcar_functional, config)
    incar = prepare_incar(recipe, patches, config)
    vasp_set = create_vasp_set(
        structure, incar, potcar_functional, config, **calc_kwargs
    )

    apply_patches(vasp_set, patches, recipe, incar)
    vasp_set.write_input(out_dir)


