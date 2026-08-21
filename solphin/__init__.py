from importlib.metadata import PackageNotFoundError, version

from solphin import (
    band_structure,
    db_fom,
    db_plots,
    dos,
    final_results,
    optics,
    pv_fom,
    spectral,
    vasp_inputs,
)

try:  # single source of truth: the version declared in pyproject.toml
    __version__ = version("solphin")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "band_structure",
    "db_fom",
    "db_plots",
    "dos",
    "final_results",
    "optics",
    "pv_fom",
    "spectral",
    "vasp_inputs",
]
