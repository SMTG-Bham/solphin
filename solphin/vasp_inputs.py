"""Generate VASP input sets from the packaged calculation recipes."""

import json
from importlib.resources import files
from pathlib import Path
from typing import Any, TypeAlias, TypedDict, cast

from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Kpoints
from pymatgen.io.vasp.sets import VaspInputSet
from sumo.io.castep import CastepCell

# A VASP INCAR tag value. Tags are scalars, except MAGMOM, which the recipes
# carry as a per-element mapping.
IncarValue: TypeAlias = str | int | float | bool | dict[str, float]

# Patches that _prepare_vdw_tags knows how to turn into dispersion tags. It
# returns {} for anything else, so this guard only exists to keep the call off
# the hot path -- but it must list rvv10, which the vdW branch also serves.
_VDW_PATCHES = ("vdw_d3_bj", "vdw_d3", "vdw_d4", "rvv10")


class RecipeConfig(TypedDict):
    """Shape of solphin/resources/base_recipes.json.

    The top level is genuinely heterogeneous -- POTCAR_FUNCTIONAL is a bare
    string while its siblings are mappings -- which is why the previous
    dict[str, dict[str, str | dict[str, str]]] could not describe it.
    """

    INCAR: dict[str, dict[str, IncarValue]]
    POTCAR_FUNCTIONAL: str
    POTCAR: dict[str, str]
    PATCHES: dict[str, dict[str, IncarValue]]


def read_structure_pmg(filename: str | Path) -> Structure:
    """Read a crystal structure file into a pymatgen Structure.

    Supported formats include POSCAR, CIF and other pymatgen-compatible
    structure files, plus CASTEP ``.cell`` files, which pymatgen cannot
    read itself and are routed through sumo instead.

    Parameters
    ----------
    filename : str or Path
        Path to the structure file.

    Returns
    -------
    Structure
        Pymatgen Structure object representing the crystal structure.
    """
    if Path(filename).suffix.lower() == ".cell":
        cell_structure: Structure = CastepCell.from_file(str(filename)).structure
        return cell_structure
    structure = Structure.from_file(filename)
    return structure


def _load_config(fname: str) -> RecipeConfig:
    """Load a JSON configuration file from the package resources.

    Parameters
    ----------
    fname : str
        Filename of the JSON configuration file in the package resource
        directory.

    Returns
    -------
    RecipeConfig
        Parsed JSON recipe configuration.
    """
    resource_path = files("solphin.resources") / fname  # adjust module name
    with resource_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def _determine_potcar_functional(
        recipe: str,
        potcar_functional: str | None,
        config: RecipeConfig,
) -> str:
    """Determine the appropriate POTCAR functional for a VASP calculation.

    Selection priority: an explicitly requested functional, then a
    recipe-specific choice (LDA), then the configuration default.

    Parameters
    ----------
    recipe : str
        Calculation recipe or exchange-correlation functional name, e.g.
        ``"LDA"``, ``"PBE"`` or ``"HSE06"``.
    potcar_functional : str or None
        Explicitly requested POTCAR functional; used directly when provided.
    config : RecipeConfig
        Recipe configuration with the default ``"POTCAR_FUNCTIONAL"``.

    Returns
    -------
    str
        Selected POTCAR functional.
    """
    if potcar_functional:
        return potcar_functional
    if recipe == "LDA":
        return "LDA_64"
    return config["POTCAR_FUNCTIONAL"]


def _prepare_incar(
        recipe: str,
        patches: list[str],
        config: RecipeConfig,
) -> dict[str, IncarValue]:
    """Prepare an INCAR dictionary from a recipe and optional patches.

    Selects the base recipe from the configuration, applies the patches, and
    adds special handling for hybrid functionals.

    Parameters
    ----------
    recipe : str
        Calculation recipe or functional type, e.g. ``"PBE"`` or ``"HSE06"``.
    patches : list of str
        Patch names that modify the INCAR settings, e.g. structural
        relaxation or spin settings.
    config : RecipeConfig
        Recipe configuration with the base INCAR templates under
        ``"INCAR"`` and patch settings under ``"PATCHES"``.

    Returns
    -------
    dict of str to IncarValue
        Final INCAR settings for the VASP calculation.
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
        incar: dict[str, IncarValue],
        potcar_functional: str,
        config: RecipeConfig,
        **calc_kwargs: Any,
) -> VaspInputSet:
    """Create a VASP input set from a structure and prepared parameters.

    Parameters
    ----------
    structure : Structure
        Atomic structure for the VASP calculation.
    incar : dict of str to IncarValue
        INCAR settings with the calculation parameters.
    potcar_functional : str
        Identifier for the POTCAR functional, e.g. ``"PBE"`` or
        ``"LDA_64"``.
    config : RecipeConfig
        Recipe configuration with the POTCAR definitions.
    **calc_kwargs
        Additional keyword arguments passed to ``VaspInputSet``.

    Returns
    -------
    VaspInputSet
        Fully constructed VASP input set ready for calculation.
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


def _prepare_vdw_tags(recipe: str, patches: list[str]) -> dict[str, int | float | bool]:
    """Generate VASP INCAR tags for van der Waals corrections.

    Supports the D3(BJ), D3, D4 and rVV10 dispersion models, with
    functional-specific parameters where required.

    Parameters
    ----------
    recipe : str
        Exchange-correlation functional or calculation recipe, e.g.
        ``"PBE"``, ``"HSE06"`` or ``"R2SCAN"``.
    patches : list of str
        Calculation patches naming the vdW correction scheme:
        ``"vdw_d3"``, ``"vdw_d3_bj"``, ``"vdw_d4"`` or ``"rvv10"``.

    Returns
    -------
    dict of str to int, float or bool
        INCAR tags for the selected vdW correction, or an empty dictionary
        when no correction is requested.
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
        incar: dict[str, IncarValue],
) -> None:
    """Apply optional calculation patches to a VASP input set in place.

    Parameters
    ----------
    vasp_set : VaspInputSet
        VASP input set to modify.
    patches : list of str
        Patch identifiers: ``"relax_cell"`` raises ENCUT for structural
        relaxation, ``"gamma_only"`` sets a Gamma-point-only k-mesh,
        ``"vdw_d3"``/``"vdw_d3_bj"``/``"vdw_d4"``/``"rvv10"`` apply
        dispersion corrections, and ``"lobster"`` raises NBANDS for orbital
        analysis.
    recipe : str
        Exchange-correlation functional or calculation type, used for the
        vdW parameter selection.
    incar : dict of str to IncarValue
        Base INCAR settings used as a reference for parameter updates.
    """
    if "relax_cell" in patches:
        # ENCUT is numeric in every recipe; IncarValue is wide only because
        # MAGMOM is a mapping and ALGO/PREC are strings.
        encut = cast(float, vasp_set.user_incar_settings.get("ENCUT", incar["ENCUT"])) * 1.3
        vasp_set.user_incar_settings["ENCUT"] = encut

    if "gamma_only" in patches:
        # VaspInputSet documents this field as "dict or Kpoints" (sets.py:153)
        # but annotates it as plain dict (sets.py:227), so the Kpoints form it
        # explicitly supports does not type-check.
        vasp_set.user_kpoints_settings = Kpoints(kpts=((1, 1, 1)))  # type: ignore[assignment]

    if any(patch in patches for patch in _VDW_PATCHES):
        vdw_tags = _prepare_vdw_tags(recipe, patches)
        vasp_set.user_incar_settings.update(vdw_tags)

    if "lobster" in patches and "NBANDS" not in vasp_set.user_incar_settings:
        vasp_set.user_incar_settings["NBANDS"] = vasp_set.estimate_nbands() * 2


def write_vasp_calculation(
        structure: Structure,
        recipe: str,
        out_dir: str | Path,
        patches: list[str] | None = None,
        potcar_functional: str | None = None,
        **calc_kwargs: Any,
) -> None:
    """Generate and write a complete VASP calculation input set to disk.

    Constructs the INCAR, POSCAR, POTCAR and KPOINTS files via the VASP
    input set framework, applying any requested patches.

    Parameters
    ----------
    structure : Structure
        Atomic structure for the calculation.
    recipe : str
        Calculation recipe or functional, e.g. ``"PBE"`` or ``"HSE06"``.
    out_dir : str or Path
        Directory the VASP input files are written into.
    patches : list of str or None, optional
        Modifications to apply, e.g. vdW corrections or relaxation
        settings. Default is None, treated as no patches.
    potcar_functional : str or None, optional
        Explicit POTCAR functional selection. Default is None, which
        chooses a default from the configuration.
    **calc_kwargs
        Additional keyword arguments passed to the VASP input set
        constructor, e.g. k-points or metadata.

    Notes
    -----
    The workflow: load the base recipe configuration, determine the POTCAR
    functional, prepare the INCAR settings, create the VASP input set,
    apply the patches, and write the final input files to disk.
    """
    patches = patches or []

    config = _load_config("base_recipes.json")
    potcar_functional = _determine_potcar_functional(recipe, potcar_functional, config)
    incar = _prepare_incar(recipe, patches, config)
    vasp_set = _create_vasp_set(
        structure, incar, potcar_functional, config, **calc_kwargs
    )

    _apply_patches(vasp_set, patches, recipe, incar)
    vasp_set.write_input(str(out_dir))
