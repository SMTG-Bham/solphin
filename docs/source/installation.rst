Installation
============

``solphin`` is not yet published to PyPI, so install it from a checkout:

.. code-block:: bash

   git clone https://github.com/SMTG-Bham/solphin
   cd solphin
   pip install -e .

Python 3.11 or newer is required (``requires-python`` in ``pyproject.toml``).

Optional extras
---------------

``plot_db_combined_interactive`` and ``plot_final_result_interactive`` drive
``%matplotlib widget``, which needs the ``ipympl`` backend:

.. code-block:: bash

   pip install -e ".[interactive]"

The other extras are ``tutorial`` (JupyterLab and the notebook stack),
``docs`` (Sphinx, the Wagtail theme and ``myst-nb``, which renders the
tutorial notebooks) and ``dev`` (pytest, ruff, mypy, pre-commit and the
packaging tools).

Development environment
-----------------------

``environment.yml`` builds a complete conda environment, including an
editable install of ``solphin``:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate solphin

Two entries come from PyPI, through the ``pip:`` block in that file, rather
than from conda-forge: ``sumo`` and ``castepxbin``. As of September 2026
conda-forge's ``sumo`` pins ``castepxbin 0.1.0`` and every conda-forge
``castepxbin`` build requires ``numpy <2``, which cannot coexist with the
``numpy>=2.0`` this package needs. The comment at the top of
``environment.yml`` has the details.

VASP pseudopotentials
---------------------

For the ``VASP`` input file generation functionality, make your ``VASP``
pseudopotentials discoverable through a ``pymatgen`` configuration file at
``$HOME/.pmgrc.yaml`` containing:

.. code-block:: yaml

   PMG_VASP_PSP_DIR: <Path to VASP pseudopotential top directory>

This applies to VASP only — CASTEP generates on-the-fly pseudopotentials, so
its input generation needs no local pseudopotential library.

CASTEP support
--------------

CASTEP support needs no extra installation: the ``.bands`` and ``.cell``
readers come with ``sumo`` and its ``castepxbin`` dependency, both installed
with ``solphin``. Optics parsing consumes an OptaDOS ``<seed>_epsilon.dat``
file — see :doc:`castep` for the workflow.
