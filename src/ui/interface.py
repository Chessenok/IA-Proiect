import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QGroupBox, QFormLayout, 
    QGraphicsView, QGraphicsScene, QTextEdit
)
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPen, QBrush, QColor, QPainter, QPolygonF, QFont
import math

from ui.bridge import executa_tsp, executa_nlp, obtine_lista_dataseturi
from utils.io_utils import incarca_date_tsp
class MapView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self) # type: ignore
        self.setScene(self.scene) # type: ignore
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.scene.setSceneRect(0, 0, 800, 600) # type: ignore
        self.setBackgroundBrush(QBrush(QColor("#2b2b2b")))
        
        self.noduri_curente = [] # Păstrăm coordonatele local pentru a desena traseul

    def draw_test_nodes(self, noduri: list[tuple[int, int]]):
        self.noduri_curente = noduri # Salvăm nodurile
        self.scene.clear() # type: ignore
        pen = QPen(QColor("#00ff00"))
        brush = QBrush(QColor("#00aa00"))
        
        for x, y in noduri:
            self.scene.addEllipse(x-5, y-5, 10, 10, pen, brush) # type: ignore

    def draw_path(self, traseu: list[int]):
        if not traseu or not self.noduri_curente:
            return

        self.scene.clear() # type: ignore
        # 1. Redesenăm toate nodurile ca bază (verzi)
        self.draw_test_nodes(self.noduri_curente)

        pen_path = QPen(QColor("#4cc9f0"), 2)
        
        # 2. Desenăm liniile și săgețile
        for i in range(len(traseu)):
            idx_start = traseu[i]
            idx_end = traseu[(i + 1) % len(traseu)]
            
            p1_coords = self.noduri_curente[idx_start]
            p2_coords = self.noduri_curente[idx_end]
            
            p1 = QPointF(p1_coords[0], p1_coords[1])
            p2 = QPointF(p2_coords[0], p2_coords[1])

            self.scene.addLine(p1.x(), p1.y(), p2.x(), p2.y(), pen_path) # type: ignore
            self.draw_arrow(p1, p2, pen_path)

        # 3. EVIDENȚIERE START: Redesenăm primul nod peste restul elementelor
        self.draw_start_node(traseu[0])

    def draw_start_node(self, start_idx: int):
        """Redesează nodul de start cu roșu intens."""
        coords = self.noduri_curente[start_idx]
        x, y = coords[0], coords[1]
        
        # Pen și Brush roșu pentru evidențiere
        pen_start = QPen(QColor("#ff0000"), 2)
        brush_start = QBrush(QColor("#ff4d4d"))
        
        # Desenăm un cerc puțin mai mare pentru a fi vizibil
        self.scene.addEllipse(x-7, y-7, 14, 14, pen_start, brush_start) # type: ignore
        
        # Opțional: Adăugăm un text mic deasupra: "START"
        text = self.scene.addText("START", QFont("Arial", 10, QFont.Bold)) # type: ignore
        text.setDefaultTextColor(QColor("#ff0000"))
        text.setPos(x - 20, y - 25)

    def draw_arrow(self, start_point, end_point, pen):
        """Calculează și desenează vârful săgeții pe segment."""
        # Dimensiunea săgeții
        arrow_size = 10
        
        # Calculăm unghiul liniei
        line = (end_point - start_point)
        angle = math.atan2(-line.y(), line.x())

        # Poziționăm săgeata la 70% din lungimea segmentului (pentru vizibilitate)
        arrow_pos = start_point + line * 0.7

        # Calculăm cele două puncte ale bazei triunghiului
        p1 = arrow_pos + QPointF(math.sin(angle + math.pi / 3) * arrow_size,
                                 math.cos(angle + math.pi / 3) * arrow_size)
        p2 = arrow_pos + QPointF(math.sin(angle + math.pi - math.pi / 3) * arrow_size,
                                 math.cos(angle + math.pi - math.pi / 3) * arrow_size)

        # Creăm poligonul pentru vârful săgeții
        arrow_head = QPolygonF([arrow_pos, p1, p2])
        
        # Desenăm săgeata plină
        self.scene.addPolygon(arrow_head, pen, QBrush(pen.color())) # type: ignore

class AISolver(QMainWindow):
    """
    Fereastra principala a aplicatiei.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TSP & NLP AI Solver - Control Panel")
        self.resize(1200, 800)

        # Widget-ul central care va tine tot
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layout-ul principal: Orizontal (Panou stanga pt setari, Panou dreapta pt harta)
        main_layout = QHBoxLayout(central_widget)

        # --- PANOU STÂNGA: CONTROALE ---
        controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(10, 10, 10, 10)
        
        # 1. Selectie Problema/Mod
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Drone Routing (TSP)", "Text Analysis (NLP)"])
        controls_layout.addWidget(QLabel("<b>Mod Operare:</b>"))
        controls_layout.addWidget(self.mode_combo)

        # 2. Selectie Algoritm TSP
        self.algo_group = QGroupBox("Configurare TSP")
        algo_layout = QFormLayout()
        
        self.algo_combo = QComboBox()
        self.algo_combo.addItems([
            "BKT (Backtracking)", 
            "NN (Nearest Neighbor)", 
            "HC (Hill Climbing)", 
            "SA (Simulated Annealing)", 
            "GA (Genetic Algorithm)"
        ])
        algo_layout.addRow("Algoritm:", self.algo_combo)

        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(obtine_lista_dataseturi())
        algo_layout.addRow("Set Date:", self.dataset_combo)

        self.algo_group.setLayout(algo_layout)
        controls_layout.addWidget(self.algo_group)

        # 3. Butoane de Actiune
        self.btn_load = QPushButton("Încarcă / Generează Hartă")
        self.btn_load.clicked.connect(self.load_data)
        
        self.btn_run = QPushButton("Rulează Algoritm")
        self.btn_run.clicked.connect(self.run_algorithm)
        # Stil simplu pentru butonul de run
        self.btn_run.setStyleSheet("background-color: #2a82da; color: white; font-weight: bold; padding: 8px;")

        controls_layout.addWidget(self.btn_load)
        controls_layout.addWidget(self.btn_run)

        # 4. Zona de Log-uri / Raportare
        controls_layout.addWidget(QLabel("<b>Status Execuție & Rezultate:</b>"))
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        controls_layout.addWidget(self.log_console)

        # --- PANOU DREAPTA: VIZUALIZARE ---
        self.map_view = MapView()

        # Asamblare finala layout-uri (Stanga ocupa ~30%, Dreapta ~70%)
        main_layout.addLayout(controls_layout, stretch=1)
        main_layout.addWidget(self.map_view, stretch=3)
    # --- METODE LOGICE ---

    def log(self, message: str):
        """Adauga mesaje in consola din stanga."""
        self.log_console.append(message)

    def load_data(self):
        # Aici vei apela generatorul tau de puncte sau vei incarca fisierul TSPLIB
        alg_ales = self.algo_combo.currentText()
        set_ales = self.dataset_combo.currentText()
        self.log(f"[*] Incarcare '{set_ales}' pentru algoritmul '{alg_ales}'...")
        matrice, noduri = incarca_date_tsp(set_ales)
        
        # Test randare
        self.map_view.draw_test_nodes(noduri)
        self.log("[+] Harta a fost generata si desenata pe canvas.")
        return matrice

    def run_algorithm(self):
        alg_ales = self.algo_combo.currentText()
        # Presupunem că ai o metodă self.matrice_curenta care stochează datele
        matrice = self.load_data()
        
        self.log(f"[*] Se executa {alg_ales}...")
        
        # --- APELUL DIN BRIDGE ---
        traseu, cost = executa_tsp(alg_ales, matrice) # type: ignore
        
        self.log(f"[+] Executie finalizata. Cost optim: {cost}")
        # Aici apelezi metoda de desenare din map_view
        self.map_view.draw_path(traseu) # type: ignore


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Fortam un stil general pentru o curatenie mai buna pe diverse OS-uri
    app.setStyle("Fusion") 
    
    window = AISolver()
    window.show()
    
    sys.exit(app.exec())