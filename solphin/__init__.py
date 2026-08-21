from importlib.metadata import PackageNotFoundError, version

import solphin.db_fom
import solphin.db_plots
import solphin.dos
import solphin.final_results
import solphin.optics
import solphin.pv_fom
import solphin.spectral
import solphin.vasp_inputs

try:  # single source of truth: the version declared in pyproject.toml
    __version__ = version("solphin")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0"
