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
pip install -e . --no-deps
```

The editable install is deliberately a separate step — see the comment at the
top of [environment.yml](environment.yml) for why `--no-deps` matters here.

Python 3.10 or newer is required: `solphin/band_structure.py` uses PEP 604
unions (`str | Path`) in annotations that are evaluated at runtime. The
development environment pins 3.11.

For the `VASP` input generation functionality you will also need your `VASP`
pseudopotentials configured through `$HOME/.pmgrc.yaml`, as described in the
[README](README.md).

## Making a change

1. Fork the repository and create a branch off `main`, named after what it
   does (e.g. `fix-slme-units`, `add-thickness-sweep`).
2. Make your change, keeping it focused — one logical change per pull request.
3. Run the checks below locally. There is currently no CI, so local checks are
   what stands between a change and `main`.
4. Open a pull request against `main`, explaining what the change does and why.
   Link any related issue. If the change affects results, include a before/after
   plot or numbers.

Please keep unrelated reformatting out of the diff; it makes review much harder.

### Style and linting

The project uses [ruff](https://docs.astral.sh/ruff/) with its default rule set:

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
* docstrings on every public function, in the `Parameters:` / `Returns:` style
  used throughout (see `solphin/db_fom.py` for an example), naming the type and
  the unit of each argument and return value.

### Tests

`pytest` and `pytest-cov` are included in the development environment. There is
no test suite yet, so new tests are especially valuable — put them in a
top-level `tests/` directory mirroring the module layout
(`tests/test_db_fom.py` and so on):

```bash
pytest
```

If you are fixing a bug, a test that fails before your fix and passes after it
is the ideal contribution. For numerical routines, prefer checking against an
analytic limit or a published value over pinning whatever the code currently
returns.

### Documentation

Documentation is built with Sphinx from `docs/source`:

```bash
make -C docs html
```

The result lands in `docs/build/html`. New public functions should be reachable
from the API pages, and anything that changes the workflow should be reflected
in the tutorial notebook.

### Adding a dependency

Runtime dependencies must be added to **both**
[pyproject.toml](pyproject.toml) and [environment.yml](environment.yml), which
are kept in sync by hand. Note that `sumo` is currently listed only in
`environment.yml` despite being imported by `solphin/dos.py` and
`solphin/band_structure.py`. Please keep new additions in both files, and say in
the pull request why the dependency is needed.

## Licence

By contributing, you agree that your contributions will be licensed under the
[MIT Licence](LICENSE) that covers this project.
