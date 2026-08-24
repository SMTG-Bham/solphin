"""Importing solphin must not reconfigure global logging or warnings.

Seven modules used to call ``logging.basicConfig``, set the root logger's level,
disable matplotlib's ``font_manager`` logger, or install ``warnings`` filters at
module top level. ``solphin/__init__.py`` imports all nine modules eagerly, so
``import solphin`` reconfigured the root logger and the warnings filters of any
program that used the library.

The property is deliberately narrow. It forbids *configuring* logging at import,
not using it: a module may add ``logger = logging.getLogger(__name__)`` and call
``logger.info(...)`` and these tests still pass.

The check runs in a fresh interpreter, for two independent reasons:

* ``conftest.py`` imports solphin at collection time, so the package is already
  imported before any test body here runs.
* pytest's logging plugin attaches a capture handler to the root logger for the
  duration of every test, so an in-process ``assert not
  logging.getLogger().handlers`` would fail no matter what solphin does.
"""

import json
import os
import subprocess
import sys
from typing import Any

import pytest

from conftest import REPO_ROOT

# Records global state either side of the import and prints one JSON line.
# Nothing is asserted in the child: the parent asserts, so a failure reports the
# values actually seen rather than a bare non-zero exit status.
_PROBE = """
import json
import logging
import warnings

def first_index(predicate):
    return next(
        (i for i, f in enumerate(warnings.filters) if predicate(f)), -1
    )


root = logging.getLogger()
baseline_level = root.level

import solphin  # noqa: F401

# Python's own defaults contain both of these, in this order. A blanket
# ``filterwarnings("ignore", category=DeprecationWarning)`` is byte-identical to
# the second, so CPython dedupes it and only its *position* changes - it jumps
# ahead of the __main__ default and takes __main__ visibility down with it.
main_default = first_index(
    lambda f: f[0] == "default" and f[2] is DeprecationWarning and f[3] == "__main__"
)
blanket_ignore = first_index(
    lambda f: f[0] == "ignore" and f[1] is None and f[2] is DeprecationWarning
)

patterned = [
    (f[0], f[1].pattern, f[2].__name__)
    for f in warnings.filters
    if f[1] is not None
    and ("layout engine" in f[1].pattern
         or "invalid value encountered in multiply" in f[1].pattern)
]

print(json.dumps({
    "baseline_level": baseline_level,
    "level": root.level,
    "handlers": [repr(h) for h in root.handlers],
    "font_manager_disabled": logging.getLogger("matplotlib.font_manager").disabled,
    "main_default_index": main_default,
    "blanket_ignore_index": blanket_ignore,
    "patterned_filters": patterned,
}))
"""


@pytest.fixture(scope="module")
def probe() -> dict[str, Any]:
    """Global state in a fresh interpreter whose only job was to import solphin."""
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        # Under -c the cwd goes on sys.path first, so this measures the working
        # copy even when a different solphin is installed in the environment.
        cwd=REPO_ROOT,
        # solphin pulls in pyplot; pin the backend so the probe cannot block on
        # a GUI toolkit on a developer machine that has DISPLAY set.
        env={**os.environ, "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"probe interpreter failed:\n{result.stderr}"
    return json.loads(result.stdout.splitlines()[-1])


def test_import_adds_no_root_handler(probe: dict[str, Any]) -> None:
    """Importing solphin leaves the root logger with no handlers attached."""
    assert probe["handlers"] == []


def test_import_leaves_root_level_alone(probe: dict[str, Any]) -> None:
    """Importing solphin leaves the root logger at whatever level it found."""
    assert probe["level"] == probe["baseline_level"]


def test_import_does_not_disable_font_manager(probe: dict[str, Any]) -> None:
    """Quietening matplotlib's font chatter is the application's call, not ours."""
    assert probe["font_manager_disabled"] is False


def test_import_adds_no_message_filters(probe: dict[str, Any]) -> None:
    """The matplotlib layout-engine and numpy multiply filters are gone.

    Asserting the filter list is *unchanged* would fail: matplotlib, pymatgen,
    sumo, IPython, scipy and numpy legitimately add filters of their own. Only
    solphin's former filters are checked for absence.
    """
    assert probe["patterned_filters"] == []


def test_import_does_not_silence_main_deprecation_warnings(probe: dict[str, Any]) -> None:
    """Python's ``__main__`` DeprecationWarning default still wins.

    Position rather than presence is the thing to assert. ``band_structure``
    used to call ``filterwarnings("ignore", category=DeprecationWarning)``, which
    is byte-identical to a filter Python already ships; CPython therefore dedupes
    it and re-inserts it at the front, ahead of the ``__main__`` default that
    keeps DeprecationWarnings visible in the program's own code. The count is
    unchanged either way, so only the ordering reveals it.
    """
    main_default = probe["main_default_index"]
    blanket_ignore = probe["blanket_ignore_index"]
    assert main_default >= 0, "Python's __main__ default filter vanished"
    # -1 means no blanket ignore exists at all, which satisfies the property.
    assert blanket_ignore == -1 or main_default < blanket_ignore
