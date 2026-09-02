Solphin
========

.. raw:: html

   <img id="diagram-light" src="_static/solphin_transparent.png" alt="solphin logo">
   <img id="diagram-dark" src="_static/solphin_white.png" style="display:none;" alt="solphin logo">

``solphin`` helps characterise candidate photovoltaic materials from
first-principles calculations. It combines detailed-balance analysis with
the photovoltaic figure of merit of A. Crovetto, the Spectroscopic Limited
Maximum Efficiency (SLME), the Blank *et al.* maximum efficiency and
optical absorption plots to build a full picture of a material's
theoretical photovoltaic performance.

From an initial crystal structure, ``solphin`` generates the ``VASP`` or
``CASTEP`` input files for each required calculation, and reads the results
of either code.

**Please note that Solphin is still in early-stage testing and development.**

.. raw:: html

   <img id="flowchart" src="_static/solphin_workflow.drawio.png"
        alt="Flowchart of the solphin workflow, from crystal structure through
        VASP or CASTEP calculations to the efficiency analysis"
        style="background:#fff;border-radius:6px;padding:8px;">

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

.. toctree::
   :maxdepth: 1

   installation
   tutorials
   castep
   api
   formalism
