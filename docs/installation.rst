Installation & Setup
====================

SPARK is distributed both as a standalone Windows installer/executable and as an open-source Python package that can run across Windows, Linux, and macOS.

.. contents:: Table of Contents
   :local:
   :depth: 2

System Requirements
-------------------

* **Operating System**: Windows 10/11 (64-bit recommended), Ubuntu 20.04+, macOS 12+
* **Processor**: x86_64 Dual-Core or higher (Quad-Core+ recommended for multi-component VarPro fits)
* **RAM**: Minimum 4 GB (8 GB+ recommended for large multi-dataset matrices)
* **Storage**: ~500 MB for executable installation or ~200 MB for Python environment
* **Display**: Minimum resolution $1280 \times 800$ (SPARK natively supports 2K/4K high-DPI monitors)

Method 1: Windows Graphical Installer (Recommended)
---------------------------------------------------

The fastest and most user-friendly way to install SPARK on Windows is using the official installer binary:

1. Download ``SPARK_Setup.exe`` from the releases section.
2. Double-click ``SPARK_Setup.exe`` to launch the Tkinter installation wizard:

   .. code-block:: text

      ┌─────────────────────────────────────────────────────────┐
      │  SPARK Setup Wizard                                    │
      ├─────────────────────────────────────────────────────────┤
      │  Destination Folder:                                    │
      │  [ C:\Users\<User>\AppData\Local\Programs\SPARK      ]  │
      │                                                         │
      │  [X] Create Desktop Shortcut                            │
      │  [X] Create Start Menu Shortcut                         │
      │  [X] Launch SPARK after installation                    │
      │                                                         │
      │                 [ Install SPARK ]                       │
      └─────────────────────────────────────────────────────────┘

3. Choose your desired destination folder (defaults to ``%LOCALAPPDATA%\Programs\SPARK``, requiring no administrative privileges).
4. Select whether you would like desktop and Start Menu shortcuts created.
5. Click **Install SPARK**. The installer will extract all internal runtime libraries, set up shortcuts with the application icon, and create an automated ``Uninstall.bat`` script for clean future removal.

Method 2: Portable Executable
-----------------------------

If you work on restricted laboratory computers without permission to write to program folders:

1. Download the standalone archive containing ``SPARK.exe``.
2. Extract the folder anywhere on your computer (e.g., Desktop or external USB drive).
3. Double-click ``SPARK.exe`` to launch immediately. No registry entries or system files are modified.

Method 3: Running from Python Source
------------------------------------

If you wish to customize SPARK, contribute algorithms, or integrate it into automated Python data pipelines:

1. **Clone the Repository**:

   .. code-block:: bash

      git clone https://github.com/AlejandroSerranoCapote/SPARK.git
      cd spark

2. **Create a Clean Virtual Environment**:

   Using Conda:

   .. code-block:: bash

      conda create -n spark_env python=3.11
      conda activate spark_env

   Using ``venv``:

   .. code-block:: bash

      python -m venv venv
      # Windows:
      venv\Scripts\activate
      # Linux / macOS:
      source venv/bin/activate

3. **Install Dependencies**:

   .. code-block:: bash

      pip install -r requirements.txt

   Key dependencies include:
   - ``numpy >= 1.24``, ``scipy >= 1.10``: Fast array operations and nonlinear optimization.
   - ``matplotlib >= 3.7``: Interactive and publication plotting.
   - ``pandas >= 2.0``: Robust delimited data parsing.
   - ``PyQt5 >= 5.15``: Graphical interface widgets and multithreading.
   - ``h5py >= 3.8``: Hierarchical project file persistence.
   - ``sympy >= 1.12``: Symbolic ODE differentiation and equation parsing.

4. **Launch SPARK**:

   .. code-block:: bash

      python run_app.py

Troubleshooting & Known Issues
------------------------------

High-DPI / Blurry Text on Windows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
SPARK enables Qt High-DPI scaling automatically upon startup:

.. code-block:: python

   QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
   QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

If your system uses mixed display scalings (e.g., 100% on external monitor, 150% on laptop), right-click ``SPARK.exe``, go to **Properties** → **Compatibility** → **Change high DPI settings**, and check **Override high DPI scaling behavior** set to **System (Enhanced)**.

PyQt5 Platform Plugin Error on Linux
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you see ``qt.qpa.plugin: Could not load the Qt platform plugin "xcb"`` on Linux headless systems or Docker:

.. code-block:: bash

   sudo apt-get install libxcb-xinerama0 libxkbcommon-x11-0 libxcb-cursor0
   # For headless execution (automated testing or script generation):
   export QT_QPA_PLATFORM=offscreen
