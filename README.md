# Solphin

***

<img width="1600" height="480" alt="solphin" src="https://github.com/user-attachments/assets/04549cb0-a768-4322-9a17-173756d84d75" />

***

[![test](https://github.com/SMTG-Bham/solphin/actions/workflows/test.yml/badge.svg)](https://github.com/SMTG-Bham/solphin/actions/workflows/test.yml)
[![docs](https://github.com/SMTG-Bham/solphin/actions/workflows/docs.yml/badge.svg)](https://github.com/SMTG-Bham/solphin/actions/workflows/docs.yml)
[![package](https://github.com/SMTG-Bham/solphin/actions/workflows/package.yml/badge.svg)](https://github.com/SMTG-Bham/solphin/actions/workflows/package.yml)
[![codecov](https://codecov.io/gh/SMTG-Bham/solphin/branch/main/graph/badge.svg)](https://codecov.io/gh/SMTG-Bham/solphin)
<a href="https://solphin.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/Docs-Read%20the%20Docs-8CA1AF?logo=readthedocs&amp;logoColor=white" alt="Documentation"></a>

`solphin` helps characterise candidate photovoltaic materials from first-principles calculations. It combines
detailed-balance analysis with the photovoltaic figure of merit of A. Crovetto, the Spectroscopic Limited Maximum
Efficiency (SLME), the Blank *et al.* maximum efficiency and optical absorption plots to build a full picture of a
material's theoretical photovoltaic performance. From an initial crystal structure, `solphin` generates the `VASP` or
`CASTEP` input files for each required calculation, and reads the results of either code.

**Please note that Solphin is still in early-stage testing and development**

Documentation, including workflow tutorials for both codes and the Python API reference, lives at
[solphin.readthedocs.io](https://solphin.readthedocs.io/en/latest/).

<img width="1076" height="596" alt="solphin_1 drawio" src="https://github.com/user-attachments/assets/0f981e7a-fcf3-4ec0-a2ef-01980a3f56ee" />

## Installation

`solphin` is not yet published to PyPI, so install it from a checkout:

```bash
git clone https://github.com/SMTG-Bham/solphin
cd solphin
pip install -e .
```

Python 3.11 or newer is required. See the
[installation docs](https://solphin.readthedocs.io/en/latest/installation.html) for the optional extras, the conda
development environment and VASP pseudopotential setup.

## CASTEP

Every VASP-facing capability has a CASTEP counterpart — see
["Using solphin with CASTEP"](https://solphin.readthedocs.io/en/latest/castep.html) for the reference, and
[tutorial/castep_workflow_tutorial.ipynb](tutorial/castep_workflow_tutorial.ipynb) for the workflow walked end to end.

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
