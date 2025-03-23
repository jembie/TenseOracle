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
html_logo = "_images/db_chair_logo.svg"
html_favicon = "_images/db_chair_logo.svg"

html_theme_options = {
    "logo": {
        "text": "Tense Oracle",
        "image_light": "_images/db_chair_logo.svg",
        "image_dark": "_images/db_chair_logo.svg",
    },
    "icon_links": [
        {
            "name": "GitHub Project",
            "url": "https://github.com/JP-SystemsX/TenseOracle",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
        {
            "name": "Research Chair",
            "url": "https://tu-dresden.de/ing/informatik/sya/db/#intro",
            "icon": "_static/db_chair_logo.svg",
            "type": "local",
        },
    ],
    "external_links": [
        {"name": "TU Dresden Database Research Chair", "url": "https://tu-dresden.de/ing/informatik/sya/db/#intro"},
    ],
}
