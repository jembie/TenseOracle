# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "TenseOracle"
copyright = "2025, Jimmy Pöhlmann"
author = "Jimmy Pöhlmann"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
from pathlib import Path
import sys

sys.path.insert(0, str(Path("..", "ThoroughOracle-Scripts").resolve()))

extensions = ["sphinx.ext.autodoc"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/db_chair_logo.png"


html_theme_options = {
    "logo": {
        "text": "Tense Oracle Project Documentation",
        "image_light": "_static/db_chair_logo.png",
        "image_dark": "_static/db_chair_logo.png",
    }
}
