import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QGroupBox, QFormLayout, 
    QGraphicsView, QGraphicsScene, QTextEdit
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPen, QBrush, QColor, QPainter

from ui.bridge import executa_tsp, executa_nlp

class MapView(QGraphicsView):
    """
    Componenta customizata pentru randarea hartii dronei.
    Aici vom desena nodurile si vom trasa/anima muchiile.
    """
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self) # type: ignore
        self.setScene(self.scene) # type: ignore
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Dimensiune initiala a canvas-ului
        self.scene.setSceneRect(0, 0, 800, 600) # type: ignore
        self.setBackgroundBrush(QBrush(QColor("#2b2b2b"))) # Un fundal dark pentru contrast

    def draw_test_nodes(self, noduri: list[tuple[int, int]]):
        """Metoda de test pentru a verifica randarea."""
        self.scene.clear() # type: ignore
        pen = QPen(QColor("#00ff00"))
        brush = QBrush(QColor("#00aa00"))
        
        for x, y in noduri:
            self.scene.addEllipse(x-5, y-5, 10, 10, pen, brush) # type: ignore

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
        self.dataset_combo.addItems(["Generare Aleatoare (N=20)", "Harta Industriala 1", "Harta Urbana TSPLIB"])
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
        
        # Test randare
        self.map_view.draw_test_nodes([(100, 100), (400, 300), (700, 100)])
        self.log("[+] Harta a fost generata si desenata pe canvas.")

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