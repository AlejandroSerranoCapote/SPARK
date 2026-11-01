Importer API (`spark.data.importer`)
====================================

.. currentmodule:: spark.data.importer

The importer module handles universal file parsing, format detection, matrix orientation alignment, and manual spreadsheet-style assignment.

Functions
---------

.. py:function:: load_data(auto_path=None, data_path=None, wl_path=None, td_path=None)

   Master data loading dispatcher.

   If ``auto_path`` is provided, attempts to parse as a combined single CSV/TXT file. If separate files are supplied via ``data_path``, ``wl_path``, and ``td_path``, dispatches to :func:`load_from_paths`.

   :param str auto_path: Path to a single combined data file.
   :param str data_path: Path to separate 2D data matrix file.
   :param str wl_path: Path to separate 1D wavelength file.
   :param str td_path: Path to separate 1D delay time file.
   :return: Tuple ``(data_matrix, wl_vector, td_vector)``.
   :rtype: tuple of numpy.ndarray
   :raises ValueError: If file formats or dimensions cannot be resolved.

.. py:function:: load_universal_file(filepath, parent_window=None)

   Attempts automated parsing of any delimited file (CSV, TXT, DAT) or NumPy dictionary (.npy). If automatic parsing encounters ambiguities or non-numeric headers, launches :class:`DataImporterDialog` for interactive user assignment.

   :param str filepath: Absolute path to the data file.
   :param QWidget parent_window: Parent Qt widget for dialog anchoring.
   :return: Tuple ``(data_matrix, wl_vector, td_vector)`` or ``(None, None, None)`` if canceled.
   :rtype: tuple

.. py:function:: read_csv_file(path)

   Fast CSV reader assuming standard spectroscopic orientation: row 0 contains delay time headers, column 0 contains wavelength coordinates, and remaining cells contain signal values.

   :param str path: File path.
   :return: Tuple ``(wl, td, data)``.
   :rtype: tuple of numpy.ndarray

.. py:function:: load_from_paths(data_path, wl_path, td_path)

   Loads dataset from three separate files. Automatically tests comma, tab, and whitespace delimiters, skips textual headers, and validates dimensionality.

   :param str data_path: Path to 2D matrix.
   :param str wl_path: Path to 1D wavelength coordinates.
   :param str td_path: Path to 1D delay time coordinates.
   :return: Tuple ``(data_arr, wl_arr, td_arr)``.
   :rtype: tuple of numpy.ndarray

Classes
-------

.. py:class:: DataImporterDialog(raw_matrix, parent=None)

   Interactive Qt spreadsheet dialog allowing users to visually assign rows and columns to wavelength, time delay, or data roles.

   :param numpy.ndarray raw_matrix: Unparsed raw matrix loaded from disk.
   :param QWidget parent: Parent Qt window.

   .. py:method:: auto_detect_layout()

      Scans cell data types to automatically find numeric boundaries and assign roles.

   .. py:method:: assign_selection(role, color, show_errors=True)

      Assigns currently selected table cells to a role (``'WL'``, ``'TD'``, or ``'DATA'``) and applies highlighting color.
