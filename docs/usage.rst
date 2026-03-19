Uso Básico
==========

Ejecutar la Aplicación
----------------------

Una vez instaladas las dependencias, puedes lanzar la interfaz gráfica de usuario (GUI) desde la terminal:

1. Asegúrate de estar dentro de la carpeta del script.
2. Ejecuta el siguiente comando:

.. code-block:: bash

   python UltrafastSpectroscopyAnalyzer.py

Crear un Ejecutable Independiente
----------------------------------

Si prefieres crear un archivo `.exe` (para Windows) o ejecutable para evitar usar la terminal cada vez, puedes usar PyInstaller:

.. code-block:: bash

   pyinstaller --onefile --noconsole --icon=icon.ico --exclude-module PyQt6 "Ultrafast Spectroscopy Analyzer.py"
