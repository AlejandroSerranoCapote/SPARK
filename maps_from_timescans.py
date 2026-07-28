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

MODULES_STYLESHEET = """
    QMainWindow, QWidget {
        background-color: #F8F9FA; 
        color: #222222;            
        font-family: "Segoe UI", Arial, sans-serif;
        font-size: 9pt;              
    }
    QGroupBox {
        border: none;
        border-top: 1px solid #D0D0D0;
        margin-top: 18px;             
        padding-top: 15px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0px 5px 0px 0px;
        font-weight: bold;
        font-size: 10pt;
        color: #3C5488; 
    }
    QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
        background-color: #FFFFFF; 
        border: 1px solid #CED4DA;
        border-radius: 4px;
        color: #212529;            
        padding: 4px 8px;                
        min-height: 22px; 
    }
    QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus {
        border: 1px solid #80BDFF; 
    }
    QPushButton {
        background-color: #E2E8F0;
        border: 1px solid #CBD5E1;
        border-radius: 4px;
        padding: 6px 12px;
        color: #0F172A;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #CBD5E1;
        border: 1px solid #94A3B8; 
    }
    QPushButton#BtnGreen {
        background-color: #10B981; 
        border: 1px solid #059669; 
        color: white; 
    }
    QPushButton#BtnGreen:hover {
        background-color: #059669; 
    }
    QListWidget {
        background-color: #FFFFFF;
        border: 1px solid #CED4DA;
        border-radius: 4px;
        padding: 5px;
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

    def initUI(self):
            """Initializes the layout, widgets, and styles of the main GUI."""
            self.setWindowTitle("SPARK - 2D Mapper (XFEL / Timescans)")
            self.setGeometry(100, 100, 750, 850)
            
            main_widget = QWidget()
            self.setCentralWidget(main_widget)
            
            # Aplicamos el estilo Premium unificado
            self.setStyleSheet(MODULES_STYLESHEET)
            
            layout = QVBoxLayout(main_widget)
            layout.setContentsMargins(30, 20, 30, 30)
            layout.setSpacing(10)
    
            # --- 1. CONFIGURATION SECTION ---
            config_group = QGroupBox("1. Mapping Configuration (In .npy)")
            grid = QGridLayout(config_group)
            grid.setSpacing(10)
            
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
            
            grid.addWidget(QLabel("Direct Signal Key:"), 2, 0)
            grid.addWidget(self.key_sig, 2, 1, 1, 3)
            
            layout.addWidget(config_group)
    
            # --- 2. ENERGIES INPUT SECTION ---
            energy_group = QGroupBox("2. Energy (eV) / Wavelength (nm) Vector")
            e_lay = QHBoxLayout(energy_group)
            e_lay.setSpacing(10)
            
            self.e_input = QLineEdit()
            self.e_input.setPlaceholderText("E.g.: 2470.5, 2475.5, 2480.0 ...") 
            self.e_input.textChanged.connect(self.validate_counts)
            e_lay.addWidget(self.e_input)
            
            btn_e = QPushButton("Import TXT")
            btn_e.clicked.connect(self.import_energies)
            e_lay.addWidget(btn_e)
            
            layout.addWidget(energy_group)
    
            # --- 3. FILE SELECTION SECTION ---
            files_group = QGroupBox("3. Kinetic Data Files (.npy)")
            f_lay = QVBoxLayout(files_group)
            f_lay.setSpacing(10)
            
            h_btn_files = QHBoxLayout()
            btn_f = QPushButton("Select .npy Files")
            btn_f.clicked.connect(self.load_files)
            
            btn_check = QPushButton("Check Units")
            btn_check.clicked.connect(self.check_units)
            
            h_btn_files.addWidget(btn_f)
            h_btn_files.addWidget(btn_check)
            h_btn_files.addStretch()
            f_lay.addLayout(h_btn_files)
            
            self.list_w = QListWidget()
            f_lay.addWidget(self.list_w)
    
            # Status label
            self.label_status = QLabel("Ready")
            self.label_status.setAlignment(Qt.AlignCenter)
            self.label_status.setStyleSheet("color: #6C757D; font-weight: bold; font-size: 10pt;")
            f_lay.addWidget(self.label_status)
            
            layout.addWidget(files_group)
            
            layout.addStretch()
    
            # --- 4. ACTIONS SECTION ---
            act_lay = QHBoxLayout()
            act_lay.setSpacing(15)
            
            self.btn_run = QPushButton("GENERATE MAP")
            self.btn_run.setObjectName("BtnGreen") # Forzamos el color verde de acción principal
            self.btn_run.setFixedHeight(40)
            self.btn_run.clicked.connect(self.generate)
            
            self.btn_save = QPushButton("SAVE MAP")
            self.btn_save.setFixedHeight(40)
            self.btn_save.setEnabled(False)
            self.btn_save.clicked.connect(self.save)
    
            self.btn_reset = QPushButton("RESET")
            self.btn_reset.setFixedHeight(40)
            self.btn_reset.clicked.connect(self.reset_app)
    
            act_lay.addWidget(self.btn_run, stretch=2) # Le damos más peso visual al botón principal
            act_lay.addWidget(self.btn_save, stretch=1)
            act_lay.addWidget(self.btn_reset, stretch=1)
            
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
            self.label_status.setStyleSheet("color: #10B981; font-weight: bold; font-size: 10pt;") # Premium Green
        else:
            self.label_status.setText(f"MISMATCH: {nf} Files / {ne} Energies")
            self.label_status.setStyleSheet("color: #EF4444; font-weight: bold; font-size: 10pt;") # Premium Red

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

