.. solphin documentation master file, created by
   sphinx-quickstart on Wed Jun 17 12:25:24 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Solphin
=========

.. raw:: html

   <img id="diagram-light" src="_static/solphin_transparent.png">
   <img id="diagram-dark" src="_static/solphin_white.png" style="display:none;">

``solphin`` is a code developed to assist with the characterisation of novel photovoltaic materials. It combines detailed balance analysis with other tools such as the photovoltaic figure of merit proposed by A. Crovetto, Spectroscopic Limited Maximum Efficiency (SLME), Blank et. al. Maximum Efficiency and optical absorption plots to provide a full picture of a material's theoretical photovoltaic efficiency. The code supports the automatic generation of ``VASP`` input files for the required calculations based on an initial crystal structure.

**Please note that Solphin is still in early stage testing and development.**

.. raw:: html

   <img id="flowchart" src="_static/solphin_workflow.drawio.png">

Citation
--------

If you use ``solphin`` in your work, please cite the following:

* Cox, P. U., Russell, P. P., Crovetto, A., Squires, A. G., & Scanlon, D. O.
  Solphin [Computer software]. https://github.com/SMTG-Bham/solphin
* Crovetto, A., 2024. A phenomenological figure of merit for photovoltaic
  materials. Journal of Physics: Energy, 6 (2), p.025009.
* Alex M. Ganose, Adam J. Jackson, David O. Scanlon. sumo: Command-line tools
  for plotting and analysis of periodic ab initio calculations. Journal of Open
  Source Software, 2018 3 (28), 717, doi:10.21105/joss.00717.

Contents
---------
.. toctree::
   :maxdepth: 1

   installation.rst
   tutorials.rst
   api.rst
   formalism.rst
   publications.rst
   changelog.rst