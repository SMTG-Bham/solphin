Installation
============

``solphin`` is not yet published to PyPI, so install it from a checkout:

.. code-block:: bash

   git clone https://github.com/SMTG-Bham/solphin
   cd solphin
   pip install -e .

Python 3.11 or newer is required. The floor is set by the dependencies rather
than by ``solphin`` itself: the current releases of ``pymatgen``, ``numpy``,
``scipy`` and ``matplotlib`` all declare ``Requires-Python >=3.11``.

Optional extras
---------------

``plot_db_combined_interactive`` and ``plot_final_result_interactive`` drive
``%matplotlib widget``, which needs the ``ipympl`` backend:

.. code-block:: bash

   pip install -e ".[interactive]"

The other extras are ``tutorial`` (JupyterLab and the notebook dependencies),
``docs`` (Sphinx and the theme) and ``dev`` (pytest, ruff, pre-commit).

Development environment
-----------------------

``environment.yml`` builds a complete conda environment:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate solphin
   pip install --no-deps sumo castepxbin
   pip install -e . --no-deps

``sumo`` is installed from PyPI rather than conda-forge on purpose. The
conda-forge build pins ``castepxbin 0.1.0.*``, which in turn requires
``numpy >=1,<2`` — that conflicts with the ``numpy>=2.0`` this package needs
and makes the environment unsolvable. PyPI's ``castepxbin`` accepts
``numpy>=1,<3``, so the pip route resolves cleanly.

VASP pseudopotentials
---------------------

For the ``VASP`` input file generation functionality, make your ``VASP``
pseudopotentials discoverable through a ``pymatgen`` configuration file at
``$HOME/.pmgrc.yaml`` containing:

.. code-block:: yaml

   PMG_VASP_PSP_DIR: <Path to VASP pseudopotential top directory>
