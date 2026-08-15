import sys
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QTableWidget, QTableWidgetItem, 
                             QFileDialog, QMessageBox, QDialog, QLabel, QTableWidgetSelectionRange)
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtCore import Qt

class DataImporterDialog(QDialog):
    """
    Origin/Igor style window. Displays raw data in a spreadsheet
    so the user can manually assign WL, TD, and the Z matrix.
    Includes smart validation and auto-detect features.
    """
    def __init__(self, raw_matrix, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manual Data Importer")
        self.resize(950, 650)
        
        self.raw_matrix = np.array(raw_matrix)
        self.WL = None
        self.TD = None
        self.data_c = None
        
        self.initUI()
        self.populate_table()
        self.update_validation()

    def initUI(self):
        layout = QVBoxLayout(self)
        
        # --- 4. Softer Error Message ---
        lbl = QLabel("<b>We couldn't automatically read this file structure.</b><br>"
                     "Please map your data below or try the Auto-Detect feature.")
        lbl.setFont(QFont("Arial", 10))
        layout.addWidget(lbl)
        
        # --- Top Buttons (Auto-Detect & Reset) ---
        h_top_btns = QHBoxLayout()
        
        self.btn_auto = QPushButton("Auto-Detect Standard Layout (Row 0: TD, Col 0: WL)")
        self.btn_auto.setStyleSheet("background-color: #607D8B; color: white; font-weight: bold; padding: 6px;")
        self.btn_auto.clicked.connect(self.auto_detect_layout)
        h_top_btns.addWidget(self.btn_auto)

        self.btn_reset = QPushButton("Clear / Reset All")
        self.btn_reset.setStyleSheet("background-color: #E53935; color: white; font-weight: bold; padding: 6px;")
        self.btn_reset.clicked.connect(self.reset_assignments)
        h_top_btns.addWidget(self.btn_reset)
        
        layout.addLayout(h_top_btns)
        # --- Assignment Buttons ---
        h_btns = QHBoxLayout()
        
        self.btn_wl = QPushButton("1. Assign selection as WL (X)")
        self.btn_wl.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 6px;")
        self.btn_wl.clicked.connect(lambda: self.assign_selection('WL', QColor("#C8E6C9")))
        
        self.btn_td = QPushButton("2. Assign selection as TD (Y)")
        self.btn_td.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 6px;")
        self.btn_td.clicked.connect(lambda: self.assign_selection('TD', QColor("#BBDEFB")))
        
        self.btn_data = QPushButton("3. Assign selection as 2D Matrix")
        self.btn_data.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold; padding: 6px;")
        self.btn_data.clicked.connect(lambda: self.assign_selection('DATA', QColor("#FFE0B2")))
        
        h_btns.addWidget(self.btn_wl)
        h_btns.addWidget(self.btn_td)
        h_btns.addWidget(self.btn_data)
        layout.addLayout(h_btns)
        
        # --- 1. Real-time Status Panel ---
        h_status = QHBoxLayout()
        self.lbl_status_wl = QLabel("WL: Not assigned")
        self.lbl_status_td = QLabel("TD: Not assigned")
        self.lbl_status_data = QLabel("Data: Not assigned")
        
        # Style status labels
        status_style = "color: #555555; font-weight: bold;"
        self.lbl_status_wl.setStyleSheet(status_style)
        self.lbl_status_td.setStyleSheet(status_style)
        self.lbl_status_data.setStyleSheet(status_style)
        self.lbl_status_wl.setAlignment(Qt.AlignCenter)
        self.lbl_status_td.setAlignment(Qt.AlignCenter)
        self.lbl_status_data.setAlignment(Qt.AlignCenter)
        
        h_status.addWidget(self.lbl_status_wl)
        h_status.addWidget(self.lbl_status_td)
        h_status.addWidget(self.lbl_status_data)
        layout.addLayout(h_status)
        
        # --- The Spreadsheet ---
        self.table = QTableWidget()
        layout.addWidget(self.table)
        
        # --- 2. Smart Validation (Confirm Button) ---
        self.btn_confirm = QPushButton("Please assign all axes")
        self.btn_confirm.setEnabled(False) # Starts disabled
        self.btn_confirm.clicked.connect(self.check_and_accept)
        layout.addWidget(self.btn_confirm)
        
    def reset_assignments(self):
        """Clears all variables and resets cell backgrounds to white."""
        self.WL = None
        self.TD = None
        self.data_c = None
        
        rows = self.table.rowCount()
        cols = self.table.columnCount()
        
        # Repaint all cells white
        for i in range(rows):
            for j in range(cols):
                self.table.item(i, j).setBackground(QColor(Qt.white))
                
        self.table.clearSelection()
        
        # Update the UI validation to lock the confirm button again
        self.update_validation()
        
    def populate_table(self):
        """Populates the QTableWidget with the raw matrix data."""
        rows, cols = self.raw_matrix.shape
        self.table.setRowCount(rows)
        self.table.setColumnCount(cols)
        
        for i in range(rows):
            for j in range(cols):
                val = str(self.raw_matrix[i, j])
                item = QTableWidgetItem(val)
                item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled) # Read-only
                self.table.setItem(i, j, item)

    def auto_detect_layout(self):
        """Automatically assigns Col 0 to WL, Row 0 to TD, stopping at non-numeric metadata."""
        rows = self.table.rowCount()
        cols = self.table.columnCount()
        
        if rows < 2 or cols < 2:
            QMessageBox.warning(self, "Warning", "Matrix is too small for auto-detection.")
            return

        # Helper function to check if a value is a valid number
        def is_numeric(val):
            try:
                float(str(val).replace(',', '.').strip())
                return True
            except (ValueError, TypeError):
                return False

        # 1. Scan for the real boundary of the data (ignore text footers)
        last_row = rows - 1
        for i in range(1, rows):
            if not is_numeric(self.raw_matrix[i, 0]):
                last_row = i - 1
                break
                
        # 2. Scan for the real boundary of the columns (ignore text on the right)
        last_col = cols - 1
        for j in range(1, cols):
            if not is_numeric(self.raw_matrix[0, j]):
                last_col = j - 1
                break

        if last_row < 1 or last_col < 1:
            QMessageBox.warning(self, "Auto-Detect Failed", "Could not detect a valid numeric structure starting at cell (1,1).")
            return

        try:
            # Clear previous colors
            for i in range(rows):
                for j in range(cols):
                    self.table.item(i, j).setBackground(QColor(Qt.white))
            
            # WL: Column 0, from row 1 to last_row
            self.table.clearSelection()
            self.table.setRangeSelected(QTableWidgetSelectionRange(1, 0, last_row, 0), True)
            self.assign_selection('WL', QColor("#C8E6C9"), show_errors=False)
            
            # TD: Row 0, from col 1 to last_col
            self.table.clearSelection()
            self.table.setRangeSelected(QTableWidgetSelectionRange(0, 1, 0, last_col), True)
            self.assign_selection('TD', QColor("#BBDEFB"), show_errors=False)
            
            # DATA: From row 1, col 1 to last_row, last_col
            self.table.clearSelection()
            self.table.setRangeSelected(QTableWidgetSelectionRange(1, 1, last_row, last_col), True)
            self.assign_selection('DATA', QColor("#FFE0B2"), show_errors=False)
            
            self.table.clearSelection()
        except Exception as e:
            QMessageBox.warning(self, "Auto-Detect Failed", f"Could not apply layout: {e}")

    def assign_selection(self, role, color, show_errors=True):
        """Captures the selected indices, extracts the data, and colors the cells."""
        ranges = self.table.selectedRanges()
        if not ranges:
            if show_errors:
                QMessageBox.warning(self, "Warning", "You haven't selected any cells.")
            return
            
        r = ranges[0]
        row_start, row_end = r.topRow(), r.bottomRow()
        col_start, col_end = r.leftColumn(), r.rightColumn()
        
        try:
            raw_selection = self.raw_matrix[row_start:row_end+1, col_start:col_end+1]
            flat_selection = raw_selection.flatten()
            clean_strings = [str(x).replace(',', '.').strip() for x in flat_selection]
            clean_numbers = pd.to_numeric(clean_strings, errors='coerce')
            clean_numbers = np.nan_to_num(clean_numbers, nan=0.0)
            selection = clean_numbers.reshape(raw_selection.shape)
            
            if role == 'WL':
                self.WL = selection.flatten()
            elif role == 'TD':
                self.TD = selection.flatten()
            elif role == 'DATA':
                self.data_c = selection
                
            for i in range(row_start, row_end + 1):
                for j in range(col_start, col_end + 1):
                    self.table.item(i, j).setBackground(color)
                    
            # Call validation to update status and unlock confirm button
            self.update_validation()
            
        except Exception as e:
            if show_errors:
                QMessageBox.critical(self, "Fatal Error", f"Unexpected error processing data:\n{e}")

    def update_validation(self):
        """Updates the status labels and enables/disables the confirm button based on dimensions."""
        wl_ok = self.WL is not None
        td_ok = self.TD is not None
        data_ok = self.data_c is not None

        # Update Status Labels
        if wl_ok:
            self.lbl_status_wl.setText(f"WL: [OK] {len(self.WL)} points")
            self.lbl_status_wl.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            self.lbl_status_wl.setText("WL: Not assigned")
            self.lbl_status_wl.setStyleSheet("color: #555555; font-weight: bold;")

        if td_ok:
            self.lbl_status_td.setText(f"TD: [OK] {len(self.TD)} points")
            self.lbl_status_td.setStyleSheet("color: #2196F3; font-weight: bold;")
        else:
            self.lbl_status_td.setText("TD: Not assigned")
            self.lbl_status_td.setStyleSheet("color: #555555; font-weight: bold;")

        if data_ok:
            self.lbl_status_data.setText(f"Data: [OK] {self.data_c.shape[0]} x {self.data_c.shape[1]}")
            self.lbl_status_data.setStyleSheet("color: #FF9800; font-weight: bold;")
        else:
            self.lbl_status_data.setText("Data: Not assigned")
            self.lbl_status_data.setStyleSheet("color: #555555; font-weight: bold;")

        # Validate Dimensions
        all_assigned = wl_ok and td_ok and data_ok
        is_valid = False

        if all_assigned:
            shape = self.data_c.shape
            len_wl = len(self.WL)
            len_td = len(self.TD)
            
            # Check if dimensions match (either standard or transposed)
            if (shape[0] == len_wl and shape[1] == len_td) or (shape[0] == len_td and shape[1] == len_wl):
                is_valid = True

        # Update Button State
        self.btn_confirm.setEnabled(is_valid)
        
        if is_valid:
            self.btn_confirm.setText("Dimensions matched. Confirm and Import")
            self.btn_confirm.setStyleSheet("background-color: #333333; color: white; height: 40px; font-weight: bold; font-size: 11pt;")
        else:
            if all_assigned:
                self.btn_confirm.setText("Error: Dimension mismatch between axes and matrix")
                self.btn_confirm.setStyleSheet("background-color: #F44336; color: white; height: 40px; font-weight: bold;")
            else:
                self.btn_confirm.setText("Please assign all axes to continue")
                self.btn_confirm.setStyleSheet("background-color: #9E9E9E; color: white; height: 40px;")

    def check_and_accept(self):
        """Final verification before accepting."""
        if self.btn_confirm.isEnabled():
            self.accept()

# ==========================================
# UNIVERSAL LOADER LOGIC
# ==========================================
def load_universal_file(filepath, parent_window=None):
    """
    Attempts to read the file automatically. If it fails or is unknown,
    it launches the spreadsheet interface.
    Returns: data_c (2D), WL (1D), TD (1D) or (None, None, None) if canceled.
    """
    try:
        # 1. IF IT IS A .NPY
        if filepath.endswith('.npy'):
            data = np.load(filepath, allow_pickle=True)
            if data.ndim == 0:
                data_dict = data.item()
                if all(k in data_dict for k in ('data_c', 'WL', 'TD')):
                    return data_dict['data_c'], data_dict['WL'], data_dict['TD']
            
            raw_matrix = np.array(data)

        # 2. IF IT IS TEXT (.txt, .csv, .dat)
        else:
            df = pd.read_csv(filepath, sep=None, header=None, engine='python')
            raw_matrix = df.fillna(0).values

    except Exception as e:
        print(f"Automatic reading failed: {e}")
        QMessageBox.critical(parent_window, "Fatal read error", f"Could not read the file at all.\nDetail: {e}")
        return None, None, None

    # 3. PLAN B: OPEN SPREADSHEET
    dialog = DataImporterDialog(raw_matrix, parent_window)
    if dialog.exec_() == QDialog.Accepted:
        return dialog.data_c, dialog.WL, dialog.TD
    else:
        return None, None, None


# ==========================================
# TEST WINDOW (MAIN)
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Universal Reader Test")
        self.resize(400, 200)
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.lbl = QLabel("Load a file (.npy, .txt, .csv, .dat)")
        self.btn_load = QPushButton("Load File")
        self.btn_load.clicked.connect(self.open_file)
        
        layout.addWidget(self.lbl)
        layout.addWidget(self.btn_load)
        self.setCentralWidget(widget)

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open data file", "", "All files (*.*)")
        if not path: return
        
        data_c, WL, TD = load_universal_file(path, parent_window=self)
        
        if data_c is not None:
            self.lbl.setText(f"Success!\nMatrix: {data_c.shape}\nWL: {len(WL)} pts\nTD: {len(TD)} pts")
            print("Z Matrix:\n", data_c)
        else:
            self.lbl.setText("Import canceled.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())