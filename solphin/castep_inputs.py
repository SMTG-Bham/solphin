"""Generate CASTEP input sets (seedname.cell / seedname.param) from the packaged recipes."""

import copy
import json
from importlib.resources import files
from pathlib import Path
from typing import TypeAlias, TypedDict

from pymatgen.core.structure import Structure
from sumo.io.castep import Block, CastepCell, Tag

# A CASTEP .param tag value. Tags are scalars; CASTEP has no MAGMOM-style
# mapping tag - initial moments are per-site annotations in the .cell instead.
CastepParamValue: TypeAlias = str | int | float | bool


class CastepPatch(TypedDict, total=False):
    """One patch: tag updates for the .param and/or the .cell file."""

    param: dict[str, CastepParamValue]
    cell: dict[str, str]


class CastepRecipeConfig(TypedDict):
    """Shape of solphin/resources/castep_recipes.json.

    Unlike the VASP recipe file there is no pseudopotential map: CASTEP
    generates on-the-fly pseudopotentials by default, so the only
    pseudopotential setting is the per-recipe ``species_pot`` library the
    hybrid recipes carry under ``CELL``.
    """

    PARAM: dict[str, dict[str, CastepParamValue]]
    CELL: dict[str, dict[str, str]]
    CELL_DEFAULTS: dict[str, str]
    SPINS: dict[str, float]
    PATCHES: dict[str, CastepPatch]


# Patches from the VASP recipe file with no CASTEP equivalent, named in the
# unsupported-patch error so the caller learns why rather than just what.
_VASP_ONLY_PATCHES = ("elastic_tensor", "rvv10", "deformation_potential", "lobster")


def _load_config(fname: str) -> CastepRecipeConfig:
    """Load a JSON configuration file from the package resources.

    Parameters
    ----------
    fname : str
        Filename of the JSON configuration file in the package resource
        directory.

    Returns
    -------
    CastepRecipeConfig
        Parsed JSON recipe configuration.
    """
    resource_path = files("solphin.resources") / fname
    with resource_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def _expand_patches(patches: list[str], config: CastepRecipeConfig) -> list[str]:
    """Expand shorthand patches and validate every name against the recipes.

    ``"defect"`` is not a patch in the JSON - it stands for ``relax_atoms``
    plus ``spin_polarised``, mirroring the VASP input generation.

    Parameters
    ----------
    patches : list of str
        Requested patch names, e.g. ``["relax_cell", "vdw_d3_bj"]``.
    config : CastepRecipeConfig
        Recipe configuration whose ``"PATCHES"`` table defines the valid
        names.

    Returns
    -------
    list of str
        Patch names with ``"defect"`` expanded, in application order.

    Raises
    ------
    ValueError
        If a patch is unknown, or is a VASP-only patch with no CASTEP
        equivalent.
    """
    expanded = []
    for patch in patches:
        if patch == "defect":
            expanded.extend(["relax_atoms", "spin_polarised"])
        else:
            expanded.append(patch)

    for patch in expanded:
        if patch in config["PATCHES"]:
            continue
        if patch in _VASP_ONLY_PATCHES:
            raise ValueError(
                f"Patch {patch!r} is VASP-only and has no CASTEP equivalent."
            )
        supported = sorted(config["PATCHES"]) + ["defect"]
        raise ValueError(f"Unknown CASTEP patch {patch!r}; supported: {supported}")

    return expanded


def _prepare_param(
        recipe: str,
        patches: list[str],
        config: CastepRecipeConfig,
) -> dict[str, CastepParamValue]:
    """Prepare a .param tag dictionary from a recipe and optional patches.

    The recipe template is deep-copied before patches are applied, so the
    supplied configuration is never mutated.

    Parameters
    ----------
    recipe : str
        Calculation recipe or functional type, e.g. ``"PBE"`` or ``"HSE06"``.
    patches : list of str
        Patch names already expanded by ``_expand_patches``.
    config : CastepRecipeConfig
        Recipe configuration with the base .param templates under
        ``"PARAM"`` and patch settings under ``"PATCHES"``.

    Returns
    -------
    dict of str to CastepParamValue
        Final .param settings for the CASTEP calculation.

    Raises
    ------
    ValueError
        If the recipe is not in the packaged configuration.
    """
    if recipe not in config["PARAM"]:
        raise ValueError(
            f"Unknown CASTEP recipe {recipe!r}; supported: {sorted(config['PARAM'])}"
        )

    param = copy.deepcopy(config["PARAM"][recipe])
    for patch in patches:
        param.update(config["PATCHES"][patch].get("param", {}))
    return param


def _prepare_cell_tags(
        recipe: str,
        patches: list[str],
        config: CastepRecipeConfig,
        kpoint_mp_grid: tuple[int, int, int] | None = None,
        kpoint_mp_spacing: float | None = None,
) -> dict[str, str]:
    """Prepare the .cell tag dictionary from the recipe, patches and k-points.

    Layering order: configuration defaults, then per-recipe cell settings,
    then patch cell settings, then the explicit k-point arguments. A k-point
    grid from any layer removes the default ``kpoint_mp_spacing``, since the
    two tags are mutually exclusive in CASTEP.

    Parameters
    ----------
    recipe : str
        Calculation recipe or functional type, e.g. ``"PBE"`` or ``"HSE06"``.
    patches : list of str
        Patch names already expanded by ``_expand_patches``.
    config : CastepRecipeConfig
        Recipe configuration with ``"CELL_DEFAULTS"`` and per-recipe
        ``"CELL"`` settings.
    kpoint_mp_grid : tuple of int or None, optional
        Explicit Monkhorst-Pack grid subdivisions ``(nx, ny, nz)``.
        Default is None, which leaves the configured k-point tags in place.
    kpoint_mp_spacing : float or None, optional
        Maximum k-point spacing in Å⁻¹ (CASTEP convention, without the 2π
        factor of VASP's KSPACING). Default is None, which keeps the
        configuration default.

    Returns
    -------
    dict of str to str
        Final .cell tag settings.

    Raises
    ------
    ValueError
        If both ``kpoint_mp_grid`` and ``kpoint_mp_spacing`` are supplied.
    """
    if kpoint_mp_grid is not None and kpoint_mp_spacing is not None:
        raise ValueError(
            "kpoint_mp_grid and kpoint_mp_spacing are mutually exclusive in CASTEP;"
            " supply at most one."
        )

    tags = dict(config["CELL_DEFAULTS"])
    tags.update(config["CELL"].get(recipe, {}))
    for patch in patches:
        tags.update(config["PATCHES"][patch].get("cell", {}))

    if kpoint_mp_spacing is not None:
        tags["kpoint_mp_spacing"] = str(kpoint_mp_spacing)
    if kpoint_mp_grid is not None:
        tags["kpoint_mp_grid"] = " ".join(str(n) for n in kpoint_mp_grid)
    if "kpoint_mp_grid" in tags:
        tags.pop("kpoint_mp_spacing", None)

    return tags


def _apply_spins(structure: Structure, cell: CastepCell, config: CastepRecipeConfig) -> float:
    """Annotate the cell's atomic positions with initial magnetic moments.

    Elements listed in the configuration's ``"SPINS"`` map gain a
    ``spin=<moment>`` annotation on their ``positions_frac`` rows, the CASTEP
    analogue of VASP's MAGMOM.

    Parameters
    ----------
    structure : Structure
        Structure the cell was built from; provides the site order.
    cell : CastepCell
        Cell object whose ``positions_frac`` block is annotated in place.
    config : CastepRecipeConfig
        Recipe configuration with the element-to-moment ``"SPINS"`` map.

    Returns
    -------
    float
        Total initial moment over all sites, for the .param ``spin`` tag.
    """
    spins = config["SPINS"]
    rows = cell.blocks["positions_frac"].values

    total = 0.0
    for site, row in zip(structure.sites, rows):
        moment = spins.get(site.species_string)
        if moment is not None:
            row.append(f"spin={moment}")
            total += moment
    return total


def _format_param(tags: dict[str, CastepParamValue]) -> str:
    """Format .param tags as CASTEP input text.

    Parameters
    ----------
    tags : dict of str to CastepParamValue
        The .param settings; booleans are rendered as ``true``/``false``.

    Returns
    -------
    str
        The .param file contents, one ``tag : value`` line per setting.
    """
    lines = []
    for tag, value in tags.items():
        if isinstance(value, bool):
            value = "true" if value else "false"
        lines.append(f"{tag: <24}: {value}")
    return "\n".join(lines) + "\n"


def write_castep_calculation(
        structure: Structure,
        recipe: str,
        out_dir: str | Path,
        patches: list[str] | None = None,
        seedname: str | None = None,
        kpoint_mp_grid: tuple[int, int, int] | None = None,
        kpoint_mp_spacing: float | None = None,
        user_param_settings: dict[str, CastepParamValue] | None = None,
        user_cell_settings: dict[str, str] | None = None,
        user_cell_blocks: dict[str, list[list[str]]] | None = None,
) -> Path:
    """Generate and write a complete CASTEP calculation input set to disk.

    Writes ``<seedname>.cell`` (structure, k-points and cell-level settings)
    and ``<seedname>.param`` (calculation settings) built from the packaged
    recipe, with any requested patches applied. CASTEP generates on-the-fly
    pseudopotentials by default, so no pseudopotential files are written; the
    hybrid recipes select a norm-conserving library via ``species_pot``.

    Parameters
    ----------
    structure : Structure
        Atomic structure for the calculation.
    recipe : str
        Calculation recipe or functional, e.g. ``"PBE"`` or ``"HSE06"``.
        ``"R2SCAN"`` maps to CASTEP's ``RSCAN`` functional.
    out_dir : str or Path
        Directory the CASTEP input files are written into.
    patches : list of str or None, optional
        Modifications to apply, e.g. vdW corrections or relaxation settings.
        Default is None, treated as no patches. The VASP-only patches
        ``elastic_tensor``, ``rvv10``, ``deformation_potential`` and
        ``lobster`` are rejected with a ValueError.
    seedname : str or None, optional
        Seed used to name the input files. Default is None, which uses the
        reduced chemical formula of the structure.
    kpoint_mp_grid : tuple of int or None, optional
        Explicit Monkhorst-Pack grid ``(nx, ny, nz)``; replaces the default
        k-point spacing. Default is None.
    kpoint_mp_spacing : float or None, optional
        Maximum k-point spacing in Å⁻¹ (no 2π factor). Default is None,
        which keeps the recipe default. Mutually exclusive with
        ``kpoint_mp_grid``.
    user_param_settings : dict or None, optional
        Additional .param settings applied after the recipe and patches, so
        they take precedence. Default is None.
    user_cell_settings : dict or None, optional
        Additional .cell tag settings applied last. Default is None.
    user_cell_blocks : dict or None, optional
        Extra .cell blocks as a mapping of block name to rows, each row a
        list of str, e.g. a ``spectral_kpoint_list``. Default is None.

    Returns
    -------
    Path
        Path of the written ``.cell`` file; the ``.param`` file sits beside
        it. Returned because the seed may be derived from the structure and
        every follow-up step (band structure setup, OptaDOS) needs it.
    """
    config = _load_config("castep_recipes.json")
    patches = _expand_patches(patches or [], config)
    param = _prepare_param(recipe, patches, config)
    cell_tags = _prepare_cell_tags(
        recipe, patches, config, kpoint_mp_grid, kpoint_mp_spacing
    )

    cell = CastepCell.from_structure(structure)

    if "spin_polarised" in patches:
        total_spin = _apply_spins(structure, cell, config)
        if total_spin:
            param.setdefault("spin", total_spin)

    if user_param_settings:
        param.update(user_param_settings)
    if user_cell_settings:
        cell_tags.update(user_cell_settings)
        if "kpoint_mp_grid" in user_cell_settings:
            cell_tags.pop("kpoint_mp_spacing", None)

    # species_pot is a block in the .cell format, not a tag: a bare library
    # name row (e.g. "NCP19") selects that pseudopotential set for all species.
    species_pot = cell_tags.pop("species_pot", None)
    if species_pot is not None:
        cell.blocks["species_pot"] = Block([[species_pot]], [""])

    for tag, value in cell_tags.items():
        cell.tags[tag] = Tag(str(value).split(), "")

    for name, rows in (user_cell_blocks or {}).items():
        cell.blocks[name] = Block(rows, [""] * len(rows))

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    seed = seedname or structure.composition.reduced_formula

    cell_file = out_path / f"{seed}.cell"
    cell.to_file(str(cell_file))
    (out_path / f"{seed}.param").write_text(_format_param(param), encoding="utf-8")

    return cell_file
