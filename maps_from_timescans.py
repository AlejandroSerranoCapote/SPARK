import sys
import numpy as np
import matplotlib.pyplot as plt
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QPushButton, QListWidget, QLineEdit, QLabel, 
                             QFileDialog, QMessageBox, QHBoxLayout, QGroupBox, QGridLayout)
from PyQt5.QtCore import Qt

# ---------------------------------------------------------------------
# GUI Styling Constants
# ---------------------------------------------------------------------

BUTTON_STYLE = """
    QPushButton {
        background-color: #6CB66C;   /* Corporate green */
        color: white;                /* White text for good contrast */
        border: 1px solid #549A54;   /* Slightly darker green border */
        border-radius: 4px;          /* Smooth rounded corners */
        padding: 6px 12px;           
        font-weight: bold;
        font-family: "Segoe UI";
        font-size: 9pt;              
    }
    QPushButton:hover {
        background-color: #5CA55C;   /* Darkens slightly on hover */
        color: white;
        border: 1px solid #468446;   /* Border darkens as well */
    }
    QPushButton:pressed {
        background-color: #4A8C4A;   /* Dark green on click */
        border: 1px solid #4A8C4A;
        color: white;
        padding-top: 7px;            /* Sink-in effect when pressed */
        padding-left: 13px;
    }
    QPushButton:disabled {
        background-color: #A0C8A0;   /* Pale, "turned off" green for inactive buttons */
        border: 1px solid #A0C8A0;
        color: #F0F0F0;              /* Slightly faded text */
    }
"""

DARK_THEME_STYLE = """
    QDialog, QWidget {
        color: #222222;            /* Dark text */
        font-family: "Segoe UI";
        font-size: 8pt;              
    }
    QGroupBox {
        border: 1px solid #C0C0C0; /* Soft gray border instead of cyan */
        border-radius: 5px;
        margin-top: 8px;             
        padding-top: 10px;
        font-weight: bold;
        color: #000000;            
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 3px;
    }
    QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
        background-color: #FFFFFF; /* White background */
        border: 1px solid #C0C0C0;
        border-radius: 4px;
        color: #000000;            /* Black text */
        padding: 2px;                
        min-height: 18px;            
    }
    QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus {
        border: 1px solid #0078D7; /* Windows blue border on focus */
    }
    QComboBox QAbstractItemView {
        background-color: #FFFFFF;
        color: #000000;
        selection-background-color: #E5F1FB; /* Light blue background on hover */
        selection-color: #000000;
        border: 1px solid #C0C0C0;
    }
    QLabel { color: #222222; }
    QCheckBox { color: #222222; }
    
    QProgressBar {
        border: 1px solid #C0C0C0;
        border-radius: 5px;
        text-align: center;
        background-color: #FFFFFF; /* White progress bar background */
        color: #222222;
        max-height: 15px;            
    }
    QProgressBar::chunk {
        background-color: #6CB66C; /* Filling chunk is corporate green */
        border-radius: 3px;
        width: 10px;
        margin: 0.5px;
    }
"""


# ---------------------------------------------------------------------
# Data Processing Class
# ---------------------------------------------------------------------

class XFELProcessor:
    """
    Handles the backend processing of X-ray Free-Electron Laser (XFEL) kinetic data.
    Extracts time arrays and specific signals from a batch of .npy files to construct 2D maps.
    """

    def process(self, file_paths, energies, keys, time_scale=1.0):
        """
        Reads data from multiple .npy files and constructs a 2D data matrix.

        Args:
            file_paths (list of str): Paths to the .npy data files.
            energies (list of float): List of energy or wavelength values corresponding to each file.
            keys (dict): Dictionary specifying the dictionary keys to look for inside the .npy files.
                         Expected keys: 'time', 'direct_sig', 'es' (Excited State), 'gs' (Ground State).
            time_scale (float, optional): Scaling factor applied to the time array. Defaults to 1.0.

        Returns:
            tuple: A tuple containing:
                - common_td (numpy.ndarray): 1D array of filtered Time Delays.
                - energies (numpy.ndarray): 1D array of Energies/Wavelengths.
                - M (numpy.ndarray): 2D data matrix containing the compiled signals.

        Raises:
            KeyError: If a required key is missing from the .npy dictionary.
        """
        temp_d = []
        common_td = None
        
        for path in file_paths:
            data = np.load(path, allow_pickle=True).item()
            try:
                # Extract and scale the time vector
                td = data[keys['time']] * time_scale
                
                # Determine how to extract the signal: use direct signal if provided, otherwise compute ES - GS
                if keys['direct_sig'].strip():
                    if keys['direct_sig'] in data:
                        sig = data[keys['direct_sig']]
                    else:
                        raise KeyError(f"The key '{keys['direct_sig']}' does not exist.")
                else:
                    sig = data[keys['es']] - data[keys['gs']]
                    
            except KeyError as e:
                raise KeyError(f"Error in {os.path.basename(path)}: {str(e)}")
            
            temp_d.append(sig)
            if common_td is None: 
                common_td = td

        # Stack the 1D signal arrays as columns in a 2D matrix
        M = np.column_stack(temp_d)
        
        # Filter out NaN values from the time vector
        mask = ~np.isnan(common_td)
        return common_td[mask], np.array(energies), M[mask]

    def analyze_units(self, file_path, time_key):
        """
        Performs a statistical analysis on the time vector of a single file to infer its physical units.

        Args:
            file_path (str): Path to the .npy file to analyze.
            time_key (str): The dictionary key used to access the time array within the file.

        Returns:
            tuple: A tuple (unit_string, description_string) detailing the inferred unit 
                   (ps or fs) and the calculated statistics.
        """
        try:
            data = np.load(file_path, allow_pickle=True).item()
            td = data[time_key]
            td = td[~np.isnan(td)]
            
            max_val = np.abs(td).max()
            step = np.mean(np.diff(np.sort(td)))
            
            # Heuristic: If max delay is < 50 and step size is very small, it's likely picoseconds
            if max_val < 50 and step < 0.5:
                return "ps (Picoseconds)", f"Max: {max_val:.2f}, Mean step: {step:.4f}"
            else:
                return "fs (Femtoseconds)", f"Max: {max_val:.1f}, Mean step: {step:.2f}"
        except Exception as e:
            return "Error", str(e)


# ---------------------------------------------------------------------
# Application GUI Class
# ---------------------------------------------------------------------

class AppWindow(QMainWindow):
    """
    Main application window built with PyQt5. 
    Provides a GUI for users to load XFEL .npy files, specify internal dictionary keys, 
    map energies, generate a 2D contour map, and save the output.
    """
    
    def __init__(self):
        super().__init__()
        self.processor = XFELProcessor()
        self.file_list = []
        self.initUI()
        self.setStyleSheet(DARK_THEME_STYLE)

    def initUI(self):
        """Initializes the layout, widgets, and styles of the main GUI."""
        self.setWindowTitle("2D Maps from timescans builder")
        self.setGeometry(100, 100, 700, 850)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # --- CONFIGURATION SECTION ---
        config_group = QGroupBox("MAPPING AND SCALE CONFIGURATION (In .npy)")
        grid = QGridLayout()
        
        self.key_time = QLineEdit("Delay_fs_TT")
        self.key_es = QLineEdit("ES")
        self.key_gs = QLineEdit("GS")
        self.key_sig = QLineEdit("")
        self.key_sig.setPlaceholderText("Optional: Diff, Intensity...") 
        self.time_scale = QLineEdit("1.0") 
        
        grid.addWidget(QLabel("Time Key:"), 0, 0)
        grid.addWidget(self.key_time, 0, 1)
        grid.addWidget(QLabel("Time Scale Factor:"), 0, 2)
        grid.addWidget(self.time_scale, 0, 3)
        
        grid.addWidget(QLabel("Excited State Key (ES):"), 1, 0)
        grid.addWidget(self.key_es, 1, 1)
        grid.addWidget(QLabel("Ground State Key (GS):"), 1, 2)
        grid.addWidget(self.key_gs, 1, 3)
        
        grid.addWidget(QLabel("<b>Direct Signal Key:</b>"), 2, 0)
        grid.addWidget(self.key_sig, 2, 1, 1, 3)
        
        config_group.setLayout(grid)
        layout.addWidget(config_group)

        # --- ENERGIES INPUT SECTION ---
        layout.addWidget(QLabel("<b>ENERGY (eV) / WAVELENGTH (nm) VECTOR:</b>"))
        e_lay = QHBoxLayout()
        self.e_input = QLineEdit()
        
        self.e_input.setPlaceholderText("E.g.: 2470.5, 2475.5, 2480.0 ...") 
        
        self.e_input.textChanged.connect(self.validate_counts)
        e_lay.addWidget(self.e_input)
        
        btn_e = QPushButton("IMPORT TXT")
        btn_e.setStyleSheet(BUTTON_STYLE)
        btn_e.clicked.connect(self.import_energies)
        e_lay.addWidget(btn_e)
        layout.addLayout(e_lay)

        # --- FILE SELECTION SECTION ---
        layout.addWidget(QLabel("<b>KINETIC DATA FILES (.npy):</b>"))
        f_lay = QHBoxLayout()
        
        btn_f = QPushButton("SELECT .NPY FILES")
        btn_f.setStyleSheet(BUTTON_STYLE)
        btn_f.clicked.connect(self.load_files)
        
        btn_check = QPushButton("CHECK UNITS")
        btn_check.clicked.connect(self.check_units)
        
        f_lay.addWidget(btn_f)
        f_lay.addWidget(btn_check)
        layout.addLayout(f_lay)
        
        self.list_w = QListWidget()
        layout.addWidget(self.list_w)

        # Status label to show matching count of energies vs files
        self.label_status = QLabel("Ready")
        self.label_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_status)

        # --- ACTIONS SECTION ---
        act_lay = QHBoxLayout()
        
        self.btn_run = QPushButton("GENERATE MAP")
        self.btn_run.setStyleSheet(BUTTON_STYLE)
        self.btn_run.clicked.connect(self.generate)
        
        self.btn_save = QPushButton("SAVE MAP")
        self.btn_save.setStyleSheet(BUTTON_STYLE)
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save)

        self.btn_reset = QPushButton("RESET")
        self.btn_reset.clicked.connect(self.reset_app)

        act_lay.addWidget(self.btn_run)
        act_lay.addWidget(self.btn_save)
        act_lay.addWidget(self.btn_reset)
        layout.addLayout(act_lay)

    def check_units(self):
        """Runs the unit heuristic on the first loaded file and displays a message box."""
        if not self.file_list:
            QMessageBox.warning(self, "Error", "Carga archivos primero.")
            return
            
        unit, desc = self.processor.analyze_units(self.file_list[0], self.key_time.text())
        QMessageBox.information(self, "Unit Analysis", f"Detección: {unit}\n{desc}")

    def reset_app(self):
        """Clears all inputs, files, and resets the application to its default state."""
        self.file_list = []
        self.list_w.clear()
        self.e_input.clear()
        self.btn_save.setEnabled(False)
        self.validate_counts()

    def validate_counts(self):
        """
        Validates if the number of manually inputted energies matches the number of loaded files.
        Updates the UI status label with color-coded feedback.
        """
        ne = len([x for x in self.e_input.text().split(',') if x.strip()])
        nf = len(self.file_list)
        
        if nf > 0 and nf == ne:
            self.label_status.setText(f"MATCH: {nf} Files")
            self.label_status.setStyleSheet("color: #00ff00; font-weight: bold;")
        else:
            self.label_status.setText(f"MISMATCH: {nf} Files / {ne} Energies")
            self.label_status.setStyleSheet("color: #ff4444; font-weight: bold;")

    def import_energies(self):
        """Opens a file dialog to read an energy vector from a text/csv file and populates the line edit."""
        path, _ = QFileDialog.getOpenFileName(self, "Load", "", "Text (*.txt *.csv *.dat)")
        if path:
            try:
                with open(path, 'r') as f:
                    content = f.read()
                
                # Sanitize content: replace commas and newlines with spaces
                content = content.replace(',', ' ').replace('\n', ' ')
                d = np.fromstring(content, sep=' ')
                self.e_input.setText(", ".join(map(str, d)))
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"No se pudo importar el archivo:\n{e}")

    def load_files(self):
        """Opens a file dialog for the user to select multiple .npy files and updates the list widget."""
        files, _ = QFileDialog.getOpenFileNames(self, "Select", "", "Numpy (*.npy)")
        if files:
            self.file_list = sorted(files)
            self.list_w.clear()
            for f in self.file_list: 
                self.list_w.addItem(os.path.basename(f))
            self.validate_counts()

    def generate(self):
        """
        Extracts inputs from the GUI, uses XFELProcessor to build the 2D matrix, 
        and plots the result using Matplotlib. Enables the Save button upon success.
        """
        try:
            es = [float(x.strip()) for x in self.e_input.text().split(',') if x.strip()]
            ks = {
                'time': self.key_time.text(), 
                'es': self.key_es.text(), 
                'gs': self.key_gs.text(), 
                'direct_sig': self.key_sig.text()
            }
            scale = float(self.time_scale.text())
            
            # Process data
            self.td, self.wl, self.m = self.processor.process(self.file_list, es, ks, scale)
            self.btn_save.setEnabled(True)
            
            # Matplotlib Plotting
            plt.style.use('default') 
            plt.figure("XFEL 2D Map", figsize=(9, 7))
            plt.pcolormesh(self.wl, self.td, self.m, shading='auto', cmap='RdBu_r')
            plt.colorbar(label='Intensity')
            plt.xlabel('Energy / WL')
            
            unit_label = "ps" if scale == 0.001 else "fs"
            plt.ylabel(f'Delay ({unit_label})')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as ex:
            QMessageBox.critical(self, "Processing Error", str(ex))

    def save(self):
        """Saves the processed 2D matrix, Wavelength array, and Time Delay array into a new .npy file."""
        path, _ = QFileDialog.getSaveFileName(self, "Save", "2D_Map_Export.npy", "Numpy (*.npy)")
        if path:
            np.save(path, {'data_c': self.m.T, 'WL': self.wl, 'TD': self.td})
            QMessageBox.information(self, "Done", "Saved successfully.")

