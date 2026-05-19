# main.py
import sys, os
from PySide6.QtWidgets import QApplication

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from ui.interface import AISolver

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = AISolver()
    window.show()
    sys.exit(app.exec())