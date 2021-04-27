"""Sphinx configuration."""
project = "KernReg"
author = "Sebastian Gsell"
copyright = f"2021, {author}"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.bibtex",
    "insegel",
]

bibtex_bibfiles = ["refs.bib"]

html_theme = "insegel"
