# main.py
import sys, os
from PySide6.QtWidgets import QApplication

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ui.interface import AISolver
from utils.io_utils import genereaza_date_tsp, salveaza_date_tsp, incarca_date_tsp




if __name__ == "__main__":
    # matrice, noduri = genereaza_date_tsp(10)
    # salveaza_date_tsp("set1.json", matrice, noduri)
    # matrice, noduri = genereaza_date_tsp(25)
    # salveaza_date_tsp("set2.json", matrice, noduri)
    # matrice, noduri = genereaza_date_tsp(50)
    # salveaza_date_tsp("set3.json", matrice, noduri)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = AISolver()
    window.show()
    sys.exit(app.exec())