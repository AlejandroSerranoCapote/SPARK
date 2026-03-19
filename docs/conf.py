import os
import sys
# Esto le dice a Sphinx que busque el código de tu app en la carpeta principal
sys.path.insert(0, os.path.abspath('..'))

# Información del proyecto
project = 'Ultrafast Spectroscopy Analyzer'
copyright = '2024, Alejandro Serrano Capote'
author = 'Alejandro Serrano Capote'

# Extensiones útiles (autodoc lee tus docstrings, napoleon entiende formato Google/NumPy)
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# El tema visual estándar de Read the Docs
html_theme = 'sphinx_rtd_theme'

# Dejamos esto vacío por ahora para que no dé error al no existir la carpeta _static
html_static_path = []
