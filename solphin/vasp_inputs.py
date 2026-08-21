import json
from importlib.resources import files

from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Kpoints
from pymatgen.io.vasp.sets import VaspInputSet


def read_structure_pmg(filename):
    """
    Reads a crystal structure file using pymatgen.

    This function loads a structure from a supported file format (e.g. POSCAR,
    CIF, or other pymatgen-compatible structure files) and returns a Structure
    object.

    Parameters:
        filename(string or Path): path to the structure file.

    Returns:
        Structure: pymatgen Structure object representing the crystal structure.
    """

    structure = Structure.from_file(filename)
    return structure


def _load_config(fname: str) -> dict[str, dict[str, str | dict[str, str]]]:
    """
    Loads a JSON configuration file from packaged package resources.

    This function reads a configuration file bundled within the package resources
    and parses it into a Python dictionary for downstream use in the application.

    Parameters:
        fname(str): filename of the JSON configuration file located in the
            package resource directory.

    Returns:
        dict: parsed JSON configuration, typically a nested dictionary structure.
    """

    resource_path = files("solphin.resources") / fname  # adjust module name
    with resource_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def _determine_potcar_functional(
        recipe: str,
        potcar_functional: str | None,
        config: dict[str, dict[str, str | dict[str, str]]],
) -> str:
    """
    Determines the appropriate POTCAR functional for a VASP calculation.

    This function selects the POTCAR functional based on user input, the chosen
    calculation recipe, or a fallback value from a configuration dictionary.

    Selection priority:
    1. User-specified POTCAR functional (if provided)
    2. Automatic choice based on recipe type (e.g., LDA)
    3. Default value from configuration file

    Parameters:
        recipe(str): calculation recipe or exchange-correlation functional name
            (e.g. "LDA", "PBE", "HSE06").
        potcar_functional(str or None): explicitly requested POTCAR functional.
            If provided, this value is used directly.
        config(dict): configuration dictionary containing default settings,
            including "POTCAR_FUNCTIONAL".

    Returns:
        str: selected POTCAR functional string.
    """

    if potcar_functional:
        return potcar_functional
    if recipe == "LDA":
        return "LDA_64"
    return config["POTCAR_FUNCTIONAL"]


def _prepare_incar(
        recipe: str,
        patches: list[str],
        config: dict[str, dict[str, str | dict[str, str]]],
) -> dict[str, str | int | float]:
    """
    Prepares an INCAR dictionary for a VASP calculation based on a recipe and optional patches.

    This function constructs the INCAR input parameters by selecting a base recipe
    from a configuration dictionary and then applying optional patches that modify
    or extend the settings. It also applies special handling for hybrid functionals.

    Parameters:
        recipe(str): calculation recipe or functional type (e.g. "PBE", "HSE06").
        patches(list[str]): list of patch names to modify the INCAR settings.
            Common patches may include structural relaxation or spin settings.
        config(dict): configuration dictionary containing:
            - base INCAR templates under config["INCAR"]
            - optional patch settings under config["PATCHES"]

    Returns:
        dict: final INCAR settings dictionary to be used in a VASP calculation.
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


def _create_vasp_set(
        structure: Structure,
        incar: dict[str, str | int | float],
        potcar_functional: str,
        config: dict[str, dict[str, str | dict[str, str]]],
        **calc_kwargs,
) -> VaspInputSet:
    """
    Creates a VASP input set from a structure and prepared calculation parameters.

    This function assembles a complete VASP input set (INCAR, POTCAR, and
    POTCAR functional selection) using a structure, pre-defined INCAR settings,
    and configuration data. Additional keyword arguments are passed directly
    to the underlying VASP input set constructor.

    Parameters:
        structure(Structure): atomic structure for the VASP calculation.
        incar(dict): INCAR settings dictionary containing calculation parameters.
        potcar_functional(str): identifier for the POTCAR functional to use
            (e.g. "PBE", "LDA_64").
        config(dict): configuration dictionary containing POTCAR definitions
            and related setup information.
        **calc_kwargs: additional keyword arguments passed to VaspInputSet.

    Returns:
        VaspInputSet: fully constructed VASP input set ready for calculation.
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


def _prepare_vdw_tags(recipe: str, patches: list[str]) -> dict[str, int | float]:
    """
    Generates VASP INCAR tags for van der Waals (vdW) corrections.

    This function selects and returns the appropriate vdW-related INCAR parameters
    based on the chosen dispersion correction scheme and exchange-correlation recipe.
    It supports multiple dispersion models including D3 (BJ), D3, D4, and rVV10.

    Parameters:
        recipe(str): exchange-correlation functional or calculation recipe
            (e.g. "PBE", "HSE06", "R2SCAN").
        patches(list[str]): list of calculation patches specifying vdW correction
            schemes (e.g. "vdw_d3", "vdw_d3_bj", "vdw_d4", "rvv10").

    Returns:
        dict: dictionary of INCAR tags required for the selected vdW correction,
            or an empty dictionary if no vdW correction is requested.
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
                    "VDW_S8": 1.19528249,
                    "VDW_A1": 0.38663183,
                    "VDW_A2": 5.19133469}
        return {"IVDW": 13}
    if "rvv10" in patches:
        if recipe == "R2SCAN":
            return {"LUSE_VDW": True,
                    "BPARAM": 11.95,
                    "CPARAM": 0.0093}
    return {}


def _apply_patches(
        vasp_set: VaspInputSet,
        patches: list[str],
        recipe: str,
        incar: dict[str, str | int | float],
) -> None:
    """
    Applies optional calculation patches to a VASP input set.

    This function modifies an existing VASP input set in-place by applying
    a set of user-defined patches. These patches can adjust energy cutoffs,
    k-point sampling, van der Waals corrections, or electronic structure
    settings such as band count.

    Parameters:
        vasp_set(VaspInputSet): VASP input set object to be modified.
        patches(list[str]): list of patch identifiers controlling modifications.
            Supported options include:
            - "relax_cell": increases ENCUT for structural relaxation
            - "gamma_only": sets Gamma-point-only k-point mesh
            - "vdw_d3", "vdw_d3_bj", "vdw_d4": apply dispersion corrections
            - "lobster": increases NBANDS for orbital analysis
        recipe(str): exchange-correlation functional or calculation type
            used for vdW parameter selection.
        incar(dict): base INCAR settings used as a reference for parameter updates.

    Returns:
        None
    """

    if "relax_cell" in patches:
        encut = vasp_set.user_incar_settings.get("ENCUT", incar["ENCUT"]) * 1.3
        vasp_set.user_incar_settings["ENCUT"] = encut

    if "gamma_only" in patches:
        vasp_set.user_kpoints_settings = Kpoints(kpts=((1, 1, 1)))

    if "vdw_d3_bj" in patches or "vdw_d3" in patches or "vdw_d4":
        vdw_tags = _prepare_vdw_tags(recipe, patches)
        vasp_set.user_incar_settings.update(vdw_tags)

    if "lobster" in patches and "NBANDS" not in vasp_set.user_incar_settings:
        vasp_set.user_incar_settings["NBANDS"] = vasp_set.estimate_nbands() * 2


def write_vasp_calculation(
        structure: Structure,
        recipe: str,
        out_dir: str,
        patches: list[str] | None = None,
        potcar_functional: str | None = None,
        **calc_kwargs,
) -> dict[str, int] | None:
    """
    Generates and writes a complete VASP calculation input set to disk.

    This function constructs all required VASP input files (INCAR, POSCAR, POTCAR,
    and KPOINTS via the VASP input set framework), applies optional calculation
    patches, and writes the final inputs to the specified output directory.

    The workflow includes:
    1. Loading base recipe configuration.
    2. Determining the appropriate POTCAR functional.
    3. Preparing the INCAR settings.
    4. Creating the VASP input set.
    5. Applying optional patches (e.g. vdW corrections, k-point adjustments).
    6. Writing the final input files to disk.

    Parameters:
        structure(Structure): atomic structure for the calculation.
        recipe(str): calculation recipe or functional (e.g. "PBE", "HSE06").
        out_dir(str): directory where VASP input files will be written.
        patches(list[str] or None): optional list of modifications to apply
            (e.g. vdW corrections, relaxation settings). Default is empty.
        potcar_functional(str or None): explicit POTCAR functional selection.
            If None, a default is chosen from configuration.
        **calc_kwargs: additional keyword arguments passed to the VASP input set
            constructor (e.g. k-points, metadata).

    Returns:
        None or dict:
            Typically returns None; may return metadata depending on VASP input
            set implementation.
    """

    patches = patches or []

    config = _load_config("base_recipes.json")
    potcar_functional = _determine_potcar_functional(recipe, potcar_functional, config)
    incar = _prepare_incar(recipe, patches, config)
    vasp_set = _create_vasp_set(
        structure, incar, potcar_functional, config, **calc_kwargs
    )

    _apply_patches(vasp_set, patches, recipe, incar)
    vasp_set.write_input(out_dir)
