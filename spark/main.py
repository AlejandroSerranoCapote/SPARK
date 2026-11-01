# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 16:25:52 2025

@author: Alejandro
"""

import os
import sys
import time

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QDialog, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox,
    QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QDoubleSpinBox, QSpinBox, QSlider, QDial,
    QFrame, QGroupBox, QRadioButton, QCheckBox, QSpacerItem, QSizePolicy, QInputDialog
)
from PyQt5.QtGui import QFont, QPalette, QColor, QDesktopServices, QIcon
from PyQt5.QtCore import Qt, QTimer, QUrl, QSize, QEvent


from spark.gui import GlobalFitPanel
from spark.gui import AppWindow as XFELWindow

from spark.gui import FLUPSAnalyzer
from spark.gui import TASAnalyzer

STYLESHEET = """
    QMainWindow {
        background-color: #F8FAFC; 
    }

    /* --- LEFT PANEL (BRANDING) --- */
    QFrame#LeftPanel {
        background-color: #0F172A; /* Deep Slate Navy */
    }
    QLabel#LeftTitle {
        color: #FFFFFF;
        font-size: 26px;
        font-weight: 900;
        font-family: "Segoe UI Black", "Arial Black", sans-serif;
        letter-spacing: 1px;
    }
    QLabel#LeftSub {
        color: #94A3B8;
        font-size: 13px;
        line-height: 1.5;
    }

    /* --- RIGHT PANEL (CONTENT) --- */
    QLabel#SectionTitle {
        color: #64748B;
        font-size: 18px;
        font-weight: bold;
        letter-spacing: 2px;
        margin-top: 5px;
    }

   
    QPushButton[cssClass="MenuCard"] {
        background-color: #FFFFFF;
        color: #0F172A;
        font-size: 14px;
        font-weight: bold;    
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        border-left: 5px solid #CBD5E1; /* Default gray border */
    }
    QPushButton[cssClass="MenuCard"]:hover {
        background-color: #F1F5F9;
        margin-left: 4px; /* Small animation on hover */
    }
    QPushButton[cssClass="MenuCard"]:pressed {
        background-color: #E2E8F0;
    }

    /* Unique colors for each module card */
    QPushButton#CardFLUPS { border-left-color: #10B981; } /* Green */
    QPushButton#CardTAS { border-left-color: #F59E0B; } /* Orange */
    QPushButton#CardGlobal { border-left-color: #3B82F6; } /* Blue */
    QPushButton#CardXFEL { border-left-color: #8B5CF6; } /* Purple */

    /* --- GITHUB BUTTON --- */
    QPushButton#GithubBtn {
        background-color: rgba(255, 255, 255, 0.05);
        color: #E2E8F0;
        font-weight: bold;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 4px;
        padding: 8px;
    }
    QPushButton#GithubBtn:hover {
        background-color: rgba(255, 255, 255, 0.15);
        border: 1px solid #FFFFFF;
        color: #FFFFFF;
    }
"""


class MainApp(QMainWindow):
    """
    Main Window (LAUNCHER DASHBOARD).
    
    Acts as the central hub to access all sub-modules of the SPARK suite 
    (FLUPS, TAS, Global Fit, and XFEL Mapper).

    Parameters
    ----------
    None
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SPARK")
        self.setMinimumSize(900, 520) 
        self.setStyleSheet(STYLESHEET) 
        self.github_url = "https://github.com/AlejandroSerranoCapote/SPARK"
        self.initUI()

    def initUI(self):
        """Initializes the layout, styling, and widget placements for the dashboard."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout: Left and Right without margins
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ==========================================
        # LEFT PANEL 
        # ==========================================
        left_panel = QFrame()
        left_panel.setObjectName("LeftPanel")
        left_panel.setFixedWidth(300) # Fixed width for the menu
        
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(35, 45, 35, 30)

        title_text = (
            "<span style='color: #FFB300; font-weight: bold;'>S</span>pectroscopy<br>"
            "<span style='color: #FFB300; font-weight: bold;'>P</span>ump-probe<br>"
            "<span style='color: #FFB300; font-weight: bold;'>A</span>nalysis and<br>"
            "<span style='color: #FFB300; font-weight: bold;'>R</span>esearch<br>"
            "<span style='color: #FFB300; font-weight: bold;'>K</span>it"
        )
        
        title = QLabel(title_text)
        title.setObjectName("LeftTitle")
        left_layout.addWidget(title)

        left_layout.addSpacing(15)

        sub = QLabel("Data processing\nsuite for time-resolved\nspectroscopy & global fitting.")
        sub.setObjectName("LeftSub")
        left_layout.addWidget(sub)

        left_layout.addStretch()

        self.btn_github = QPushButton("Source on GitHub")
        self.btn_github.setObjectName("GithubBtn")
        self.btn_github.setCursor(Qt.PointingHandCursor)
        self.btn_github.clicked.connect(self.open_github)
        left_layout.addWidget(self.btn_github)

        author = QLabel("v2.0.0 \n© A. Serrano Capote")
        author.setStyleSheet("color: #475569; font-size: 11px; margin-top: 10px;")
        left_layout.addWidget(author)

        main_layout.addWidget(left_panel)

        # ==========================================
        # RIGHT PANEL (MODULE CARDS)
        # ==========================================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(50, 45, 50, 45)
        right_layout.setSpacing(15)

        # --- SECTION A: Experimental Techniques ---
        lbl_cat1 = QLabel("PRIMARY PROCESSING MODULES")
        lbl_cat1.setObjectName("SectionTitle")
        right_layout.addWidget(lbl_cat1)
        
        grid1 = QGridLayout()
        grid1.setSpacing(15)
        self.btn_flups = self.create_card("FLUPS ANALYZER", "Fluorescence Upconversion Data", "CardFLUPS")
        self.btn_tas   = self.create_card("TAS ANALYZER", "Transient Absorption Data", "CardTAS")
        grid1.addWidget(self.btn_flups, 0, 0)
        grid1.addWidget(self.btn_tas, 0, 1)
        right_layout.addLayout(grid1)
        
        right_layout.addSpacing(20)

        # --- SECTION B: Advanced Tools ---
        lbl_cat2 = QLabel("ADVANCED ANALYSIS & VISUALIZATION")
        lbl_cat2.setObjectName("SectionTitle")
        right_layout.addWidget(lbl_cat2)
        
        grid2 = QGridLayout()
        grid2.setSpacing(15)
        self.btn_fit = self.create_card("GLOBAL FIT", "Target Analysis & Kinetic Modeling", "CardGlobal")
        self.btn_xfel  = self.create_card("2D MAPPER", "Create 2D maps using kinetics\n (XFEL utility)", "CardXFEL")
        grid2.addWidget(self.btn_fit, 0, 0)
        grid2.addWidget(self.btn_xfel, 0, 1)
        right_layout.addLayout(grid2)

        right_layout.addStretch()
        main_layout.addWidget(right_panel)

        # Button connections
        self.btn_flups.clicked.connect(self.launch_flups)
        self.btn_tas.clicked.connect(self.launch_tas)
        self.btn_fit.clicked.connect(self.launch_global)
        self.btn_xfel.clicked.connect(self.launch_xfel)

    def create_card(self, title_text, sub_text, object_name):
        """
        Creates a module card using internal QLabels to support rich text formatting.

        Parameters
        ----------
        title_text : str
            The main title of the module card.
        sub_text : str
            The subtitle or brief description of the module.
        object_name : str
            The internal object name used for applying specific CSS styles.

        Returns
        -------
        QPushButton
            The configured interactive card widget.
        """
        btn = QPushButton()
        btn.setProperty("cssClass", "MenuCard") 
        btn.setObjectName(object_name)          
        btn.setCursor(Qt.PointingHandCursor)
        btn.setMinimumHeight(85)
        
        # Insert an INTERNAL layout to the button
        vbox = QVBoxLayout(btn)
        vbox.setContentsMargins(20, 15, 20, 15)
        vbox.setSpacing(4)
        
        # Create Title label
        title = QLabel(title_text)
        title.setStyleSheet("color: #0F172A; font-size: 15px; font-weight: bold; border: none; background: transparent;")
        title.setAttribute(Qt.WA_TransparentForMouseEvents) # Click passes through to the button
        
        # Create Subtitle label
        sub = QLabel(sub_text)
        sub.setStyleSheet("color: #64748B; font-size: 12px; font-weight: normal; border: none; background: transparent;")
        sub.setAttribute(Qt.WA_TransparentForMouseEvents)
        
        # Add texts to the button
        vbox.addWidget(title)
        vbox.addWidget(sub)
        vbox.addStretch() # Pushes the text up smoothly
        
        return btn


    def open_github(self):
        """Opens the repository URL in the user's default web browser."""
        if hasattr(self, 'github_url'):
            QDesktopServices.openUrl(QUrl(self.github_url))

    def open_tool(self, tool_window):
        """
        Hides the main menu, opens the selected tool, and ensures the menu 
        reappears when the child tool window is closed.
        
        Parameters
        ----------
        tool_window : QMainWindow or QWidget
            The specific tool instance to open.
        """
        self.current_tool = tool_window
        
        # Store the original close event of the tool window
        original_close = tool_window.closeEvent
        
        def on_close_tool(event):
            # Show the main menu again when the tool closes
            self.show()             
            # Execute the tool's standard close event
            original_close(event)  
            
        # Override the tool's close event with our custom wrapper
        tool_window.closeEvent = on_close_tool
        
        tool_window.show()
        self.hide()
        
    def launch_xfel(self):
        """Instantiates and launches the XFEL 2D Mapper tool."""
        window = XFELWindow() 
        self.open_tool(window)

    def launch_flups(self):
        """Instantiates and launches the FLUPS Analyzer tool."""
        window = FLUPSAnalyzer()
        self.open_tool(window)

    def launch_tas(self):
        """Instantiates and launches the TAS Analyzer tool."""
        window = TASAnalyzer()
        self.open_tool(window)

    def launch_global(self):
        """Instantiates and launches the Global Fit tool."""
        window = GlobalFitPanel()
        self.open_tool(window)
        

def main():
    """
    Application Entry Point for console script execution.
    
    Sets up the high-DPI scaling, applies the global application style, 
    and initializes the main event loop.
    """
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    app.setStyleSheet(STYLESHEET) 

    if sys.platform == 'win32':
        try:
            import ctypes
            myappid = 'spectroscopy.analyzer.v1'
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except Exception:
            pass

    base_path = getattr(sys, '_MEIPASS', os.path.dirname(__file__))
    candidates = [
        os.path.join(base_path, "icon.ico"),
        os.path.join(base_path, "spark", "icon.ico"),
        os.path.join(os.path.dirname(__file__), "icon.ico"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "icon.ico"),
        "icon.ico"
    ]
    for c in candidates:
        if os.path.exists(c):
            app.setWindowIcon(QIcon(c))
            break

    window = MainApp()
    window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
