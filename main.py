# main.py
import sys
import os

import matplotlib

matplotlib.use("Agg")

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import nlp.classifier  # noqa: F401
from PySide6.QtWidgets import QApplication

from ui.interface import AISolver
from utils.io_utils import genereaza_date_tsp, salveaza_date_tsp, incarca_date_tsp


if __name__ == "__main__":


    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = AISolver()
    window.show()
    sys.exit(app.exec())
