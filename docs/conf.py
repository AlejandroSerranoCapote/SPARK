import os
import sys
# Esto le dice a Sphinx que busque el código de tu app en la carpeta principal
sys.path.insert(0, os.path.abspath('..'))

# --- Información del proyecto ---
project = 'Ultrafast Spectroscopy Analyzer'
copyright = '2024, Alejandro Serrano Capote'
author = 'Alejandro Serrano Capote'
# Puedes poner la versión actual de tu software aquí
release = '0.1.0'

# --- Configuración General ---
extensions = [
    'sphinx.ext.autodoc',      # Extrae documentación de los docstrings
    'sphinx.ext.viewcode',     # Añade enlaces al código fuente
    'sphinx.ext.napoleon',     # Soporta estilos de comentarios Google/NumPy
    'sphinx.ext.autosummary',  # Genera resúmenes de API automáticamente
    'sphinx_copybutton',       # Añade un botón de "copiar" a los bloques de código
]

# Configuración de autodoc
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}
autosummary_generate = True # Generar resúmenes automáticamente

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
language = 'es' # Idioma de la interfaz de Sphinx

# --- Configuración del Tema Visual ---
html_theme = 'sphinx_rtd_theme'

# ¡AQUÍ ESTÁ TU LOGO Y FAVICON!
# Asegúrate de haber subido los archivos a docs/_static/
html_logo = '_static/icon.png'    # Logo principal en la barra lateral
html_favicon = '_static/icon.ico' # Icono pequeño de la pestaña

# Ruta para archivos estáticos (como css, js, imágenes)
html_static_path = ['_static']

# Opciones adicionales del tema (opcional)
html_theme_options = {
    'logo_only': False,       # Si es True, solo muestra el logo, no el título
    'display_version': True,   # Muestra la versión debajo del logo
}
