from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QMessageBox

class ToastNotification(QWidget):
    """
    Non-blocking floating notification with Fade-In and Fade-Out animations.
    (Corrected version to avoid transparent background issues on Windows).

    Parameters
    ----------
    text : str
        The message to be displayed in the notification.
    parent : QWidget, optional
        The parent widget over which the notification will float. Defaults to None.
    duration : int, optional
        The time in milliseconds the notification stays visible before fading out. 
        Defaults to 3000 ms (3 seconds).
    """
    def __init__(self, text, parent=None, duration=3000):
        super().__init__(parent)
        # Set as a floating frameless window that does not block clicks
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents) 
        
        # Wrap the QLabel in a layout so CSS renders correctly
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.label = QLabel(text)
        self.label.setAlignment(Qt.AlignCenter)
        
        # "Apple" style: dark, semi-transparent, rounded corners
        self.label.setStyleSheet("""
            QLabel {
                background-color: rgba(30, 33, 40, 230);
                color: #FFFFFF;
                border-radius: 8px;
                padding: 10px 24px;
                font-family: "Segoe UI", Arial, sans-serif;
                font-size: 10pt;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.label)
        self.adjustSize()
        
        self.duration = duration
        self.opacity_anim = QPropertyAnimation(self, b"windowOpacity")
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.hide_toast)

    def show_toast(self):
        """
        Calculates the geometry of the parent window and triggers the Fade-In 
        animation to display the notification.
        """
        # Position at the bottom center of the parent window
        if self.parent():
            parent_rect = self.parent().rect()
            parent_pos = self.parent().mapToGlobal(parent_rect.topLeft())
            x = parent_pos.x() + (parent_rect.width() - self.width()) // 2
            # Raise it by 70px so it floats cleanly above the progress bar
            y = parent_pos.y() + parent_rect.height() - self.height() - 70 
            self.move(x, y)
        
        self.setWindowOpacity(0.0)
        self.show()
        
        # Entry animation (Fade In)
        self.opacity_anim.setDuration(300)
        self.opacity_anim.setStartValue(0.0)
        self.opacity_anim.setEndValue(1.0)
        self.opacity_anim.start()
        
        # Start the countdown to disappear
        self.timer.start(self.duration)

    def hide_toast(self):
        """
        Triggers the Fade-Out animation and destroys the widget upon completion.
        """
        # Exit animation (Fade Out)
        self.opacity_anim.setDuration(400)
        self.opacity_anim.setStartValue(self.windowOpacity())
        self.opacity_anim.setEndValue(0.0)
        self.opacity_anim.finished.connect(self.close) # Destroy when finished
        self.opacity_anim.start()        
        
def show_error_popup(parent, title, message):
    """
    Displays a standardized critical error pop-up dialog.

    Parameters
    ----------
    parent : QWidget
        The parent widget over which the dialog will appear.
    title : str
        The title of the dialog window.
    message : str
        The error message to display.
    """
    QMessageBox.critical(parent, title, message)

def show_warning_popup(parent, title, message):
    """
    Displays a standardized warning pop-up dialog.

    Parameters
    ----------
    parent : QWidget
        The parent widget over which the dialog will appear.
    title : str
        The title of the dialog window.
    message : str
        The warning message to display.
    """
    QMessageBox.warning(parent, title, message)