# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Solphin'
copyright = '2026, Philippa U. Cox, Peter P. Russell'
author = 'Philippa U. Cox, Peter P. Russell'
release = '0.0.1'

html_static_path = ["_static"]

html_theme_options = dict(
    project_name = "",
    logo = "solphin_white.png",
    logo_alt = "solphin logo",
    logo_height = 120,
    logo_url = "",
    logo_width = 368,
    footer_links = "",
    github_url = "",
)

html_favicon = "_static/solphin_favicon.png"

html_show_sphinx = False

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_wagtail_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "myst_nb",  # for jupyter notebooks
]

myst_enable_extensions = [
    "html_admonition",
    "html_image",  # to parse html syntax to insert images
    "dollarmath",  # "amsmath", # to parse Latex-style math
]

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
}

# -- Options for nb extension -----------------------------------------------
nb_execution_mode = "off"
# nb_render_image_options = {"height": "300",}  # Reduce plots size
# myst_render_markdown_format = "gfm"
myst_heading_anchors = 2


# ignore non-consecutive level header warnings, and attempted image editing:
suppress_warnings = ["myst.header", "mystnb.image"]

templates_path = ['_templates']
exclude_patterns = []

import os
import sys

sys.path.insert(
    0,
    os.path.abspath("../../"),
)

autosummary_generate = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_wagtail_theme"
html_static_path = ['_static']
html_js_files = [
    "theme-toggle.js",
]