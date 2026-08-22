# Contributing to Solphin

Thanks for your interest in `solphin`. The code is in early-stage testing and
development, so bug reports, documentation fixes and new features are all
welcome.

By participating in this project you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to contribute

### Reporting a bug

Open an issue at
[github.com/SMTG-Bham/solphin/issues](https://github.com/SMTG-Bham/solphin/issues).
A useful report includes:

* what you expected to happen and what happened instead, with the full
  traceback if there was one;
* the smallest snippet that reproduces the problem;
* your `solphin` version (`python -c "import solphin; print(solphin.__version__)"`),
  Python version and OS;
* for problems with parsing or plotting `VASP` output, the relevant input and
  output files (`vasprun.xml`, `INCAR`, `POSCAR`, ...), or a description of the
  calculation if the files are too large to attach.

### Suggesting a feature

Open an issue describing the physics or workflow you want to support and how
you would expect to call it. For anything substantial, please do this before
writing code — it is much easier to agree on an approach up front than to
rework a finished pull request.

### Asking a question

Questions about using the code are also welcome as issues. Please check the
[tutorial notebook](tutorial/full_workflow_tutorial.ipynb) and the
documentation first.

## Development setup

The conda environment file installs every runtime, docs and development
dependency:

```bash
conda env create -f environment.yml
```

```bash
conda activate solphin
```

```bash
pip install --no-deps sumo castepxbin
```

```bash
pip install -e . --no-deps
```

Both pip steps are deliberately separate from the conda solve — see the comment
at the top of [environment.yml](environment.yml) for why `--no-deps` matters,
and why `sumo` cannot come from conda-forge.

Python 3.11 or newer is required. `solphin/band_structure.py` uses PEP 604
unions (`str | Path`) in annotations that are evaluated at runtime, which would
allow 3.10, but the current releases of `pymatgen`, `numpy`, `scipy` and
`matplotlib` all declare `Requires-Python >=3.11`. The development environment
pins 3.11.

For the `VASP` input generation functionality you will also need your `VASP`
pseudopotentials configured through `$HOME/.pmgrc.yaml`, as described in the
[README](README.md).

## Making a change

1. Fork the repository and create a branch off `main`, named after what it
   does (e.g. `fix-slme-units`, `add-thickness-sweep`).
2. Make your change, keeping it focused — one logical change per pull request.
3. Run the checks below locally. CI runs the same checks on every pull request
   — lint, types and the test matrix in `.github/workflows/test.yml`, the
   documentation build in `.github/workflows/docs.yml`, and the packaging checks
   in `.github/workflows/package.yml` — so running them first saves a round
   trip.
4. Open a pull request against `main`, explaining what the change does and why.
   Link any related issue. If the change affects results, include a before/after
   plot or numbers.

Please keep unrelated reformatting out of the diff; it makes review much harder.

### Style and linting

The project uses [ruff](https://docs.astral.sh/ruff/). The lint rules cover
import hygiene and docstring conventions (`E402`, `F401`, `F811`, `I`, `D`
under the numpy convention) — see `[tool.ruff.lint]` in
[pyproject.toml](pyproject.toml). `ruff check` is what CI enforces:

```bash
ruff check .
```

```bash
ruff format .
```

Beyond that, follow the conventions already in the package:

* module-level constants for physical quantities pulled from
  `scipy.constants`, with a comment giving the unit;
* explicit units in variable names or comments wherever a quantity is
  dimensional — this is a physics code and unit errors are the easiest bugs to
  introduce and the hardest to spot;
* docstrings on every module and function, in strict
  [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) format
  (see `solphin/db_fom.py` for examples), naming the type and the unit of each
  argument and return value. The one-line summary starts on the same line as
  the opening `"""`, in the imperative mood; parameter entries are
  `name : type`, with a `, optional` marker and a ``Default is `x`.`` sentence
  for defaulted arguments:

  ```python
  def voc(E_gap: float, photon_spectrum: NDArray, Tcell: float) -> float:
      """Calculate the open-circuit voltage.

      Parameters
      ----------
      E_gap : float
          Optical band gap in eV.
      photon_spectrum : numpy.ndarray
          Converted photon flux spectrum from ``convert_spectrum``.
      Tcell : float
          Operating temperature of the cell in K.

      Returns
      -------
      float
          Open-circuit voltage in V.
      """
  ```

  `ruff check` enforces the layout (`D` rules, numpy convention).

### Tests

`pytest` and `pytest-cov` are included in the development environment. The
suite lives in the top-level `tests/` directory and mirrors the module layout
(`tests/test_db_fom.py` and so on). Read [tests/README.md](tests/README.md)
first — it explains what is anchored to analytic limits and documents the ten
`xfail(strict=True)` markers that stand in for known defects, which means
**fixing one of those defects turns the suite red** and is your cue to drop the
marker:

```bash
pytest
```

If you are fixing a bug, a test that fails before your fix and passes after it
is the ideal contribution. For numerical routines, prefer checking against an
analytic limit or a published value over pinning whatever the code currently
returns.

CI measures coverage on every run and reports it to
[Codecov](https://codecov.io/gh/SMTG-Bham/solphin), which comments on the pull
request. The whole suite currently covers about 85% of `solphin/`; the patch
check asks for 80% on the lines you add or change, so new code should arrive
with tests. To see the same numbers before you push — bare `--cov` reads its
settings from `[tool.coverage.run]` in [pyproject.toml](pyproject.toml):

```bash
pytest --cov --cov-report=term-missing
```

Add `--cov-report=html` for a line-by-line report in `htmlcov/`. Both that
directory and the `coverage.xml` CI uploads are already gitignored.

### Documentation

Documentation is built with Sphinx from `docs/source`:

```bash
make -C docs html
```

The result lands in `docs/build/html`. New public functions should be reachable
from the API pages, and anything that changes the workflow should be reflected
in the tutorial notebook.

CI builds the same documentation with warnings promoted to errors, so a
malformed docstring fails the build rather than reaching the published pages.
The build is warning-free today; keep it that way by running it the way CI does
before you push:

```bash
make -C docs html SPHINXOPTS="-W"
```

### Adding a dependency

Runtime dependencies must be added to **both**
[pyproject.toml](pyproject.toml) and [environment.yml](environment.yml), which
are kept in sync by hand. `environment.yml` carries one deliberate asymmetry:
`sumo` is a runtime dependency in `pyproject.toml` but is absent from the conda
list, because conda-forge's build pins `castepxbin 0.1.0.*` and so caps
`numpy<2`. It is installed from PyPI instead, as a documented `--no-deps` step.
Please keep new additions in both files, and say in the pull request why the
dependency is needed.

### Adding a resource file

Anything new under `solphin/resources/` needs listing in **both**
[MANIFEST.in](MANIFEST.in), which controls the sdist, and
`[tool.setuptools.package-data]` in [pyproject.toml](pyproject.toml), which
controls the wheel. Missing one is easy to do and hard to notice: the package
reads these files through `importlib.resources` at run time, so a wheel without
them still installs and imports perfectly and only fails when someone calls the
function. `.github/workflows/package.yml` builds both artefacts and fails if a
tracked resource is absent from either, but it is cheaper to get right first
time.

## Licence

By contributing, you agree that your contributions will be licensed under the
[MIT Licence](LICENSE) that covers this project.
