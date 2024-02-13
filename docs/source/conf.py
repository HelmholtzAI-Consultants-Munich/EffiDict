# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import os
import sys
from pathlib import Path

# Add parent dir to known paths
p = Path(__file__).parents[2]
sys.path.insert(0, os.path.abspath(p))

project = "Effidict"
copyright = "2024, Helmholtz AI"
author = "Isra Mekki"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_logo = "_static/logo_effidict.png"
html_favicon = "_static/logo_effidict.png"

html_title = "Effidict"
html_theme_options = {
    "logo_only": True,
    "repository_url": "https://github.com/HelmholtzAI-Consultants-Munich/EffiDict",
    "use_repository_button": True,
    "use_fullscreen_button": True,
    "use_issues_button": True,
}

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True,
}
