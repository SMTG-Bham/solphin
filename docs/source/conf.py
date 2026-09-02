"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import sys

# make the package importable from a source checkout (autodoc + version lookup)
sys.path.insert(0, os.path.abspath("../../"))

import solphin

release = solphin.__version__

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Solphin'
copyright = '2026, Philippa U. Cox, Peter P. Russell, Andrea Crovetto, Alexander G. Squires and David O. Scanlon'
author = 'Philippa U. Cox, Peter P. Russell, Andrea Crovetto, Alexander G. Squires, David O. Scanlon'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_wagtail_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_nb",  # for jupyter notebooks
]

# Docstrings are strict numpydoc (see CONTRIBUTING.md) and, as CONTRIBUTING
# requires, name the type and unit of every argument themselves. autodoc's
# default of "signature" would render the annotation as well, so each parameter
# would carry its type twice. "description" merges the annotation into the
# existing Parameters entry -- and where the docstring already gives a type,
# the docstring's type wins.
autodoc_typehints = "description"

# The package is numpy-format only; turning the Google parser off stops
# napoleon from ever mis-reading an indented block as a Google section.
napoleon_google_docstring = False
napoleon_numpy_docstring = True  # the default, kept explicit alongside the line above

# Render Attributes sections as a variables list. The default attribute
# directives would collide with the entries autodoc already generates for the
# annotated dataclass fields ("duplicate object description" warnings).
napoleon_use_ivar = True

myst_enable_extensions = [
    "html_admonition",
    "html_image",  # to parse html syntax to insert images
    "dollarmath",  # "amsmath", # to parse Latex-style math
]

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
}

exclude_patterns = []

# -- Options for nb extension -----------------------------------------------
nb_execution_mode = "off"
myst_heading_anchors = 2

# ignore non-consecutive level header warnings, and attempted image editing:
suppress_warnings = ["myst.header", "mystnb.image"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_wagtail_theme"
html_static_path = ["_static"]

html_theme_options = dict(
    project_name="",
    logo="solphin_white.png",
    logo_alt="solphin logo",
    logo_height=120,
    logo_url="",
    logo_width=368,
    header_links="GitHub|https://github.com/SMTG-Bham/solphin",
    footer_links="",
    # Prefix for the theme's per-page "Edit on GitHub" button (pagename plus
    # source suffix is appended), so it must point into docs/source/ and keep
    # the trailing slash.
    github_url="https://github.com/SMTG-Bham/solphin/blob/main/docs/source/",
)

html_favicon = "_static/solphin_favicon.png"

html_show_sphinx = False

html_js_files = [
    # Must match the filename in _static/ exactly: Sphinx emits the <script>
    # tag for whatever is named here without checking that the file exists, so
    # a typo is a silent 404 rather than a build error.
    "theme_toggle.js",
]
