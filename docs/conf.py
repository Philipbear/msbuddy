import os
import sys

sys.path.insert(0, os.path.abspath('../msbuddy'))

# -- Project information -----------------------------------------------------

project = 'msbuddy'  # Your project name
author = 'Shipei Xing'  # Author's name
copyright = '2023, Shipei Xing'


# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',      # For auto-generating documentation from docstrings
    'sphinx.ext.viewcode',     # Links to source code
    'sphinx.ext.napoleon',     # Support for NumPy and Google style docstrings
    'sphinx.ext.autosummary',  # Generate autodoc summaries
    'sphinx.ext.todo',         # Support for todo items
    'sphinx.ext.coverage',     # Support for coverage checks
    'sphinx.ext.githubpages',  # Support for hosting docs on GitHub
    'sphinx.ext.autosectionlabel',  # Support for referencing sections with its title
    'sphinx_rtd_theme',        # Read the Docs theme
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'  # Read the Docs theme
html_theme_options = {
    'description': 'Bottom-up MS/MS interrogation-based molecular formula annotation for mass spectrometry data.',
    'github_user': 'Philipbear',
    'github_repo': 'msbuddy',
    'github_button': True,
    'github_banner': True,
    'collapse_navigation': False,
    'display_version': True,
    'navigation_depth': 3,
    'includehidden': True,
    'titles_only': False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
