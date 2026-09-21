# Solphin

***

<img width="1600" height="480" alt="solphin" src="https://github.com/user-attachments/assets/04549cb0-a768-4322-9a17-173756d84d75" />

***

[![test](https://github.com/SMTG-Bham/solphin/actions/workflows/test.yml/badge.svg)](https://github.com/SMTG-Bham/solphin/actions/workflows/test.yml)
[![docs](https://github.com/SMTG-Bham/solphin/actions/workflows/docs.yml/badge.svg)](https://github.com/SMTG-Bham/solphin/actions/workflows/docs.yml)
[![package](https://github.com/SMTG-Bham/solphin/actions/workflows/package.yml/badge.svg)](https://github.com/SMTG-Bham/solphin/actions/workflows/package.yml)
[![PyPI](https://img.shields.io/pypi/v/solphin?logo=pypi&logoColor=white)](https://pypi.org/project/solphin/)
[![codecov](https://codecov.io/gh/SMTG-Bham/solphin/branch/main/graph/badge.svg)](https://codecov.io/gh/SMTG-Bham/solphin)
<a href="https://solphin.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/Docs-Read%20the%20Docs-8CA1AF?logo=readthedocs&amp;logoColor=white" alt="Documentation"></a>

`solphin` helps characterise candidate photovoltaic materials from first-principles calculations. It combines
detailed-balance analysis with the photovoltaic figure of merit of A. Crovetto, the Spectroscopic Limited Maximum
Efficiency (SLME), the Blank *et al.* maximum efficiency and optical absorption plots to build a full picture of a
material's theoretical photovoltaic performance. From an initial crystal structure, `solphin` generates the `VASP` or
`CASTEP` input files for each required calculation, and reads the results of either code.


Documentation, including workflow tutorials for both codes and the Python API reference, lives at
[solphin.readthedocs.io](https://solphin.readthedocs.io/en/latest/).

<img width="1076" height="596" alt="solphin_1 drawio" src="https://github.com/user-attachments/assets/0f981e7a-fcf3-4ec0-a2ef-01980a3f56ee" />

## Installation

```bash
pip install solphin
```

Python 3.11 or newer is required.

To work on `solphin` itself, or to run the tutorial notebooks from the repository, install from a checkout instead:

```bash
git clone https://github.com/SMTG-Bham/solphin
cd solphin
pip install -e .
```

See the [installation docs](https://solphin.readthedocs.io/en/latest/installation.html) for the optional extras, the
conda development environment and VASP pseudopotential setup.


## Quick start

The detailed-balance analysis needs nothing beyond the package itself, so it is the quickest way to check an install.
This computes the Shockley-Queisser limit for a 1.39 eV absorber (the Cu<sub>2</sub>GeS<sub>3</sub> band gap used in
the tutorials) under the bundled AM1.5G spectrum, then draws the three-panel detailed-balance figure:

```python
import matplotlib.pyplot as plt

import solphin.db_fom as db_fom
import solphin.db_plots as db_plots

E_gap = 1.39   # band gap in eV
Tcell = 300.0  # cell temperature in K

spectrum = db_fom.load_spectrum("AM1.5")
photon_spectrum = db_fom.convert_spectrum(spectrum)

print(f"Jsc = {db_fom.jsc(E_gap, photon_spectrum, Tcell):.1f} A m^-2")
print(f"Voc = {db_fom.voc(E_gap, photon_spectrum, Tcell):.3f} V")
print(f"SQ efficiency limit = {100 * db_fom.max_eff(E_gap, photon_spectrum, Tcell):.1f} %")

db_plots.plot_db_combined(spectrum=photon_spectrum, Egap=E_gap, Tcell=Tcell, spectrum_type="AM1.5")
plt.savefig("detailed_balance.png", dpi=150, bbox_inches="tight")
```

The full workflow, from generating `VASP` or `CASTEP` inputs for a crystal structure to reading the finished
calculations back into band gap, effective mass, absorption, SLME and figure-of-merit analyses, is walked through in
the tutorial notebooks:
[full_workflow_tutorial.ipynb](https://github.com/SMTG-Bham/solphin/blob/main/tutorial/full_workflow_tutorial.ipynb)
for `VASP` and
[castep_workflow_tutorial.ipynb](https://github.com/SMTG-Bham/solphin/blob/main/tutorial/castep_workflow_tutorial.ipynb)
for `CASTEP`. The same notebooks are rendered in the
[documentation](https://solphin.readthedocs.io/en/latest/tutorials.html), alongside the
[API reference](https://solphin.readthedocs.io/en/latest/api.html).

## Contributing and support

Bug reports, feature requests and questions about using the code all go through the
[issue tracker](https://github.com/SMTG-Bham/solphin/issues). For a bug, please include the smallest snippet that
reproduces it, the full traceback and your `solphin` version
(`python -c "import solphin; print(solphin.__version__)"`).

If you would like to contribute code or documentation,
[CONTRIBUTING.md](https://github.com/SMTG-Bham/solphin/blob/main/CONTRIBUTING.md) covers the development environment,
the style and docstring conventions, how to run the tests and how to build the documentation. This project follows the
[Contributor Covenant](https://github.com/SMTG-Bham/solphin/blob/main/CODE_OF_CONDUCT.md) code of conduct.

## Citation

If you use `solphin` in your work, please cite the following:

* Cox, P. U., Russell, P. P., Crovetto, A., Squires, A. G., Slocombe, L, & Scanlon, D. O.
  Solphin [Computer software]. https://github.com/SMTG-Bham/solphin
* Crovetto, A., 2024. A phenomenological figure of merit for photovoltaic materials. Journal of Physics: Energy, 6 (2),
  p.025009.
* Alex M. Ganose, Adam J. Jackson, David O. Scanlon. sumo: Command-line tools for plotting and analysis of periodic ab
  initio calculations. Journal of Open Source Software, 2018 3 (28), 717, doi:10.21105/joss.00717.

## Acknowledgements

The developers Philippa U. Cox, Peter P. Russell and Louie Slocombe would like to thank Alexander G. Squires, Andrea Crovetto and David
O. Scanlon for their guidance on this project, Brooke Busbee for her work on the branding and Jacob Baggott for his
assistance.

## License

`solphin` is released under the [MIT License](https://github.com/SMTG-Bham/solphin/blob/main/LICENSE).
