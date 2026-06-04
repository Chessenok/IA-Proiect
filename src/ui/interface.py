import sys
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from nlp.classifier import (
    DEMO_DATASET,
    MODEL_REGISTRY,
    TextClassifierRunner,
    preprocess_steps,
)
from nlp.analyzer import (
    analyze_sentiment,
    detect_domain_terms,
    extract_keywords,
    format_nlp_result,
)
from nlp.dictionaries import load_all_dictionaries
from PySide6.QtCore import Qt, QPointF, QTimer
from PySide6.QtGui import QPen, QBrush, QColor, QPainter, QPolygonF, QFont, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QHeaderView,
    QInputDialog
)

import time

from ui.bridge import executa_tsp, obtine_lista_dataseturi
from utils.benchmark import SessionBenchmark, render_benchmark_chart_png
from utils.io_utils import incarca_date_tsp, genereaza_date_tsp, salveaza_date_tsp
from utils.metrics import save_performance_plot_or_fallback


class MapView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)  # type: ignore
        self.setScene(self.scene)  # type: ignore
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.scene.setSceneRect(0, 0, 800, 600)  # type: ignore
        self.setBackgroundBrush(QBrush(QColor("#2b2b2b")))

        self.noduri_curente = []

    def draw_test_nodes(self, noduri: list[tuple[int, int]]):
        self.noduri_curente = noduri
        self.scene.clear()  # type: ignore
        pen = QPen(QColor("#00ff00"))
        brush = QBrush(QColor("#00aa00"))

        for x, y in noduri:
            self.scene.addEllipse(x - 5, y - 5, 10, 10, pen, brush)  # type: ignore

    def draw_path(self, traseu: list[int]):
        if not traseu or not self.noduri_curente:
            return

        self.scene.clear()  # type: ignore
        self.draw_test_nodes(self.noduri_curente)

        pen_path = QPen(QColor("#4cc9f0"), 2)

        for i in range(len(traseu)):
            idx_start = traseu[i]
            idx_end = traseu[(i + 1) % len(traseu)]

            p1_coords = self.noduri_curente[idx_start]
            p2_coords = self.noduri_curente[idx_end]

            p1 = QPointF(p1_coords[0], p1_coords[1])
            p2 = QPointF(p2_coords[0], p2_coords[1])

            self.scene.addLine(p1.x(), p1.y(), p2.x(), p2.y(), pen_path)  # type: ignore
            self.draw_arrow(p1, p2, pen_path)

        self.draw_start_node(traseu[0])

    def draw_start_node(self, start_idx: int):
        coords = self.noduri_curente[start_idx]
        x, y = coords[0], coords[1]

        pen_start = QPen(QColor("#ff0000"), 2)
        brush_start = QBrush(QColor("#ff4d4d"))

        self.scene.addEllipse(x - 7, y - 7, 14, 14, pen_start, brush_start)  # type: ignore

        text = self.scene.addText("START", QFont("Arial", 10, QFont.Bold))  # type: ignore
        text.setDefaultTextColor(QColor("#ff0000"))
        text.setPos(x - 20, y - 25)

    def draw_arrow(self, start_point, end_point, pen):
        arrow_size = 10
        line = end_point - start_point
        angle = math.atan2(-line.y(), line.x())
        arrow_pos = start_point + line * 0.7

        p1 = arrow_pos + QPointF(
            math.sin(angle + math.pi / 3) * arrow_size,
            math.cos(angle + math.pi / 3) * arrow_size,
        )
        p2 = arrow_pos + QPointF(
            math.sin(angle + math.pi - math.pi / 3) * arrow_size,
            math.cos(angle + math.pi - math.pi / 3) * arrow_size,
        )

        arrow_head = QPolygonF([arrow_pos, p1, p2])
        self.scene.addPolygon(arrow_head, pen, QBrush(pen.color()))  # type: ignore


class PlotPanel(QLabel):
    """Afișează PNG-ul graficului de performanță; reîncarcă la redimensionare."""

    def __init__(self):
        super().__init__("Grafic performanță (după rulare)")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(220)
        self.setMinimumWidth(320)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setStyleSheet("background-color: #1e1e1e; color: #888;")
        self._plot_path: Path | None = None

    def clear_plot(self):
        self._plot_path = None
        self.clear() # Șterge imaginea (QPixmap-ul) din QLabel
        self.setText("Grafic performanță (după rulare)")

    def set_plot(self, path: Path | str) -> bool:
        self._plot_path = Path(path).resolve()
        return self._apply_pixmap()

    def _apply_pixmap(self) -> bool:
        if self._plot_path is None or not self._plot_path.is_file():
            return False

        pixmap = QPixmap(str(self._plot_path))
        if pixmap.isNull():
            return False

        self.setText("")
        w = max(self.width(), 320)
        h = max(self.height(), 180)
        scaled = pixmap.scaled(
            w,
            h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)
        return True

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._plot_path is not None:
            self._apply_pixmap()

    def showEvent(self, event):
        super().showEvent(event)
        if self._plot_path is not None:
            QTimer.singleShot(0, self._apply_pixmap)


class BenchmarkPanel(QWidget):
    """Tabel + grafic cost vs timp pentru toate rulările din sesiune."""

    def __init__(self):
        super().__init__()
        self.session = SessionBenchmark()
        self._chart_path: Path | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        header = QHBoxLayout()
        header.addWidget(QLabel("<b>Benchmark (cost vs timp)</b>"))
        header.addStretch()
        btn_clear = QPushButton("Curăță")
        btn_clear.setFixedWidth(72)
        btn_clear.clicked.connect(self.clear)
        header.addWidget(btn_clear)
        layout.addLayout(header)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["#", "Alg.", "Set date", "Cost", "Timp (s)", "Parametri"]
        )
        self.table.horizontalHeader().setSectionResizeMode(
            5, QHeaderView.ResizeMode.Stretch
        )
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setMaximumHeight(140)
        layout.addWidget(self.table)

        self.chart_label = QLabel("Rulează algoritmi pentru comparare")
        self.chart_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.chart_label.setMinimumHeight(160)
        self.chart_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.chart_label.setStyleSheet("background-color: #1e1e1e; color: #888;")
        layout.addWidget(self.chart_label)

    def clear(self) -> None:
        self.session.clear()
        self._remove_chart_file()
        self.table.setRowCount(0)
        self.chart_label.clear()
        self.chart_label.setText("Rulează algoritmi pentru comparare")

    def record_run(
        self,
        algorithm: str,
        dataset: str,
        cost: float,
        elapsed_sec: float,
        params: dict | None = None,
    ) -> None:
        self.session.add_run(algorithm, dataset, cost, elapsed_sec, params)
        self.refresh()

    def refresh(self) -> None:
        self._fill_table()
        self._update_chart()

    def _fill_table(self) -> None:
        runs = self.session.runs
        self.table.setRowCount(len(runs))
        for row, entry in enumerate(runs):
            cells = [
                str(entry.run_id),
                entry.algorithm_short(),
                entry.dataset,
                f"{entry.cost:.2f}",
                f"{entry.elapsed_sec:.3f}",
                entry.params_summary(),
            ]
            for col, text in enumerate(cells):
                self.table.setItem(row, col, QTableWidgetItem(text))

    def _remove_chart_file(self) -> None:
        if self._chart_path and self._chart_path.is_file():
            self._chart_path.unlink(missing_ok=True)
        self._chart_path = None

    def _update_chart(self) -> None:
        self._remove_chart_file()
        if not self.session.runs:
            self.chart_label.clear()
            self.chart_label.setText("Rulează algoritmi pentru comparare")
            return

        path = render_benchmark_chart_png(self.session.runs)
        if path is None:
            return

        self._chart_path = path
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            return

        self.chart_label.setText("")
        w = max(self.chart_label.width(), 280)
        h = max(self.chart_label.height(), 160)
        self.chart_label.setPixmap(
            pixmap.scaled(
                w,
                h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.session.runs:
            self._update_chart()


class AISolver(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TSP & NLP AI Solver - Control Panel")
        self.resize(1400, 850)
        self.text_classifier = TextClassifierRunner()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(10, 10, 10, 10)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Drone Routing (TSP)", "Text Analysis (NLP)"])
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        controls_layout.addWidget(QLabel("<b>Mod Operare:</b>"))
        controls_layout.addWidget(self.mode_combo)

        self.nlp_group = self._build_nlp_controls()
        controls_layout.addWidget(self.nlp_group)

        self.algo_group = QGroupBox("Configurare TSP")
        algo_layout = QFormLayout()

        self.algo_combo = QComboBox()
        self.algo_combo.addItems([
            "BKT (Backtracking)",
            "NN (Nearest Neighbor)",
            "HC (Hill Climbing)",
            "SA (Simulated Annealing)",
            "GA (Genetic Algorithm)",
        ])
        algo_layout.addRow("Algoritm:", self.algo_combo)

        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(obtine_lista_dataseturi())
        algo_layout.addRow("Set Date:", self.dataset_combo)

        self.algo_group.setLayout(algo_layout)
        controls_layout.addWidget(self.algo_group)

        self.sa_group = self._build_sa_params()
        controls_layout.addWidget(self.sa_group)

        self.ga_group = self._build_ga_params()
        controls_layout.addWidget(self.ga_group)

        self.algo_combo.currentTextChanged.connect(self._on_algo_changed)
        self._on_algo_changed(self.algo_combo.currentText())

        self.btn_load = QPushButton("Încarcă / Generează Hartă")
        self.btn_load.clicked.connect(self.load_data)

        self.btn_generate = QPushButton("Generează set nou")
        self.btn_generate.clicked.connect(self.generate_new_dataset)

        self.btn_run = QPushButton("Rulează Algoritm")
        self.btn_run.clicked.connect(self.run_algorithm)
        self.btn_run.setStyleSheet(
            "background-color: #2a82da; color: white; font-weight: bold; padding: 8px;"
        )

        controls_layout.addWidget(self.btn_load)
        controls_layout.addWidget(self.btn_generate)
        controls_layout.addWidget(self.btn_run)

        self.btn_team = QPushButton("Informații echipă")
        self.btn_team.clicked.connect(self._show_team_info)
        controls_layout.addWidget(self.btn_team)

        controls_layout.addWidget(QLabel("<b>Status Execuție & Rezultate:</b>"))
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        controls_layout.addWidget(self.log_console)

        self.map_view = MapView()

        self.plot_panel = PlotPanel()
        self.benchmark_panel = BenchmarkPanel()

        charts_splitter = QSplitter(Qt.Orientation.Horizontal)
        charts_splitter.addWidget(self.plot_panel)
        charts_splitter.addWidget(self.benchmark_panel)
        charts_splitter.setStretchFactor(0, 3)
        charts_splitter.setStretchFactor(1, 2)
        charts_splitter.setSizes([520, 380])

        viz_splitter = QSplitter(Qt.Orientation.Vertical)
        viz_splitter.addWidget(self.map_view)
        viz_splitter.addWidget(charts_splitter)
        viz_splitter.setStretchFactor(0, 3)
        viz_splitter.setStretchFactor(1, 2)
        viz_splitter.setSizes([480, 300])

        self.nlp_input = QTextEdit()
        self.nlp_input.setPlaceholderText(
            "Introduceți un text nou pentru clasificare..."
        )
        self.nlp_input.setText(
            "Echipa a lansat un model nou de inteligență artificială pentru "
            "procesarea documentelor și automatizarea fluxurilor de lucru."
        )

        self.nlp_output = QTextEdit()
        self.nlp_output.setReadOnly(True)
        self.nlp_output.setPlaceholderText("Rezultatele clasificării vor apărea aici...")

        nlp_layout = QVBoxLayout()
        nlp_layout.addWidget(QLabel("<b>Text nou pentru predicție:</b>"))
        nlp_layout.addWidget(self.nlp_input, stretch=2)
        nlp_layout.addWidget(QLabel("<b>Rezultat clasificator text:</b>"))
        nlp_layout.addWidget(self.nlp_output, stretch=3)
        nlp_widget = QWidget()
        nlp_widget.setLayout(nlp_layout)

        self.viz_stack = QStackedWidget()
        self.viz_stack.addWidget(viz_splitter)
        self.viz_stack.addWidget(nlp_widget)

        main_layout.addLayout(controls_layout, stretch=1)
        main_layout.addWidget(self.viz_stack, stretch=4)

        self._on_mode_changed(self.mode_combo.currentText())

    def _build_nlp_controls(self) -> QGroupBox:
        group = QGroupBox("Clasificare Text ML")
        layout = QFormLayout()

        self.nlp_dataset_label = QLabel("Demo local")
        self.nlp_dataset_label.setWordWrap(True)
        layout.addRow("Dataset:", self.nlp_dataset_label)

        btn_demo = QPushButton("Încarcă demo")
        btn_demo.clicked.connect(lambda: self.load_nlp_dataset(DEMO_DATASET))
        layout.addRow(btn_demo)

        btn_file = QPushButton("Alege JSON/CSV")
        btn_file.clicked.connect(self.choose_nlp_dataset)
        layout.addRow(btn_file)

        self.nlp_test_pct = QSpinBox()
        self.nlp_test_pct.setRange(10, 40)
        self.nlp_test_pct.setSingleStep(5)
        self.nlp_test_pct.setValue(20)
        layout.addRow("Test (%):", self.nlp_test_pct)

        self.nlp_stop_words = QComboBox()
        self.nlp_stop_words.addItems(["english", "romanian", "none"])
        self.nlp_stop_words.setCurrentText("romanian")
        layout.addRow("Stop words:", self.nlp_stop_words)

        self.nlp_ngram = QComboBox()
        self.nlp_ngram.addItems(["Unigrame (1,1)", "Bigrame (1,2)", "Trigrame (1,3)"])
        layout.addRow("ngram_range:", self.nlp_ngram)

        self.nlp_max_features = QSpinBox()
        self.nlp_max_features.setRange(100, 10000)
        self.nlp_max_features.setSingleStep(100)
        self.nlp_max_features.setValue(3000)
        layout.addRow("max_features:", self.nlp_max_features)

        self.nlp_sublinear_tf = QCheckBox("sublinear_tf")
        self.nlp_sublinear_tf.setChecked(True)
        layout.addRow(self.nlp_sublinear_tf)

        self.nlp_sentiment_dict = QComboBox()
        layout.addRow("Dicționar sentiment:", self.nlp_sentiment_dict)

        self.nlp_keyword_dict = QComboBox()
        layout.addRow("Stopwords cuvinte cheie:", self.nlp_keyword_dict)

        self.nlp_domain_dict = QComboBox()
        layout.addRow("Dicționar domeniu:", self.nlp_domain_dict)

        btn_refresh_dicts = QPushButton("Reîncarcă dicționare")
        btn_refresh_dicts.clicked.connect(self._refresh_nlp_dictionary_choices)
        layout.addRow(btn_refresh_dicts)

        self.nlp_model_checks: list[QCheckBox] = []
        for model_name in MODEL_REGISTRY:
            check = QCheckBox(model_name)
            check.setChecked(True)
            self.nlp_model_checks.append(check)
            layout.addRow(check)

        self._refresh_nlp_dictionary_choices()

        group.setLayout(layout)
        return group

    def _refresh_nlp_dictionary_choices(self) -> None:
        load_all_dictionaries.cache_clear()
        dicts = load_all_dictionaries()
        sentiment = [
            name
            for name, data in dicts.items()
            if "positive" in data and "negative" in data
        ]
        stopwords = [name for name, data in dicts.items() if "words" in data]
        domain = [name for name, data in dicts.items() if "categories" in data]
        domain = ["none"] + domain

        def fill(combo: QComboBox, names: list[str], preferred: str) -> None:
            current = combo.currentText()
            combo.clear()
            combo.addItems(names)
            if current in names:
                combo.setCurrentText(current)
            elif preferred in names:
                combo.setCurrentText(preferred)

        fill(self.nlp_sentiment_dict, sentiment, "sentiment_ro_extended")
        fill(self.nlp_keyword_dict, stopwords, "stopwords_ro_extended")
        fill(self.nlp_domain_dict, domain, "drone_civil_terms")

    def _show_team_info(self) -> None:
        QMessageBox.information(
            self,
            "Informații echipă",
            "Membrii echipei:\n\n"
            "Smeșnoi Maxim - sgr. 3133A\n"
            "Duma Marian-Ștefan - sgr. 3133A\n"
            "Marula Alexandru-Ionuț - sgr. 3134B",
        )

    def _on_mode_changed(self, mode: str) -> None:
        is_tsp = "TSP" in mode
        is_nlp = not is_tsp

        self.algo_group.setVisible(is_tsp)
        self.sa_group.setVisible(is_tsp and "SA" in self.algo_combo.currentText())
        self.ga_group.setVisible(is_tsp and "GA" in self.algo_combo.currentText())
        self.nlp_group.setVisible(is_nlp)

        self.btn_load.setVisible(is_tsp)
        self.btn_generate.setVisible(is_tsp)
        self.viz_stack.setCurrentIndex(0 if is_tsp else 1)

        if is_tsp:
            self.btn_run.setText("Rulează Algoritm")
            self.btn_run.clicked.disconnect()
            self.btn_run.clicked.connect(self.run_algorithm)
        else:
            self.btn_run.setText("Antrenează și clasifică")
            self.btn_run.clicked.disconnect()
            self.btn_run.clicked.connect(self.run_nlp)
            if self.text_classifier.dataset is None:
                self.load_nlp_dataset(DEMO_DATASET)

    def _build_sa_params(self) -> QGroupBox:
        group = QGroupBox("Parametri SA")
        layout = QFormLayout()

        self.sa_tmax = QSpinBox()
        self.sa_tmax.setRange(1, 1_000_000)
        self.sa_tmax.setValue(10000)
        layout.addRow("Tmax:", self.sa_tmax)

        self.sa_tmin = QSpinBox()
        self.sa_tmin.setRange(1, 100_000)
        self.sa_tmin.setValue(1)
        layout.addRow("Tmin:", self.sa_tmin)

        self.sa_steps = QSpinBox()
        self.sa_steps.setRange(100, 1_000_000)
        self.sa_steps.setSingleStep(1000)
        self.sa_steps.setValue(50000)
        layout.addRow("Steps:", self.sa_steps)

        self.sa_updates = QSpinBox()
        self.sa_updates.setRange(10, 10_000)
        self.sa_updates.setValue(100)
        layout.addRow("Updates:", self.sa_updates)

        group.setLayout(layout)
        return group

    def _build_ga_params(self) -> QGroupBox:
        group = QGroupBox("Parametri GA")
        layout = QFormLayout()

        self.ga_pop_size = QSpinBox()
        self.ga_pop_size.setRange(10, 500)
        self.ga_pop_size.setValue(50)
        layout.addRow("Populație:", self.ga_pop_size)

        self.ga_generations = QSpinBox()
        self.ga_generations.setRange(10, 10_000)
        self.ga_generations.setValue(200)
        layout.addRow("Generații:", self.ga_generations)

        self.ga_mutation = QDoubleSpinBox()
        self.ga_mutation.setRange(0.0, 1.0)
        self.ga_mutation.setSingleStep(0.05)
        self.ga_mutation.setDecimals(2)
        self.ga_mutation.setValue(0.2)
        layout.addRow("Rată mutație:", self.ga_mutation)

        self.ga_selection_k = QSpinBox()
        self.ga_selection_k.setRange(2, 20)
        self.ga_selection_k.setValue(3)
        layout.addRow("Selection k:", self.ga_selection_k)

        group.setLayout(layout)
        return group

    def _on_algo_changed(self, algo_text: str):
        if "TSP" not in self.mode_combo.currentText():
            return
        is_sa = "SA" in algo_text
        is_ga = "GA" in algo_text
        self.sa_group.setVisible(is_sa)
        self.ga_group.setVisible(is_ga)

    def _collect_algo_params(self) -> dict:
        algo = self.algo_combo.currentText()
        if "SA" in algo:
            return {
                "Tmax": self.sa_tmax.value(),
                "Tmin": self.sa_tmin.value(),
                "steps": self.sa_steps.value(),
                "updates": self.sa_updates.value(),
            }
        if "GA" in algo:
            return {
                "pop_size": self.ga_pop_size.value(),
                "generations": self.ga_generations.value(),
                "mutation_rate": self.ga_mutation.value(),
                "selection_k": self.ga_selection_k.value(),
            }
        return {}

    def log(self, message: str):
        self.log_console.append(message)
    
    def generate_new_dataset(self):
        # 1. Cere numărul de noduri de la utilizator
        n, ok = QInputDialog.getInt(
            self, 
            "Generare set de date TSP", 
            "Introduceți numărul de orașe (noduri):", 
            20,  # Valoare default
            5,   # Minim
            500, # Maxim
            1    # Step
        )
        
        if not ok:
            return # Utilizatorul a dat Cancel
            
        self.log(f"[*] Generare set de date nou cu {n} noduri...")
        
        # 2. Generează coordonatele și matricea
        # Folosim dimensiunile map_view-ului pentru a ne asigura că încap pe ecran
        latime = int(self.map_view.sceneRect().width())
        inaltime = int(self.map_view.sceneRect().height())
        matrice, noduri = genereaza_date_tsp(n, latime, inaltime)
        
        # 3. Creăm un nume unic de fișier pe baza timestamp-ului
        filename = f"random_{n}_noduri_{int(time.time())}.json"
        
        try:
            # 4. Salvăm fișierul
            salveaza_date_tsp(filename, matrice, noduri)
            self.log(f"[+] Setul de date a fost salvat ca '{filename}'.")
            
            # 5. Adăugăm noul fișier în combobox și îl selectăm automat
            self.dataset_combo.addItem(filename)
            self.dataset_combo.setCurrentText(filename)
            
            # 6. Încărcăm și desenăm direct noua hartă pe ecran
            self.load_data()
            
        except Exception as e:
            self.log(f"[-] Eroare la salvarea setului de date: {str(e)}")

    def load_data(self):
        alg_ales = self.algo_combo.currentText()
        set_ales = self.dataset_combo.currentText()
        self.log(f"[*] Incarcare '{set_ales}' pentru algoritmul '{alg_ales}'...")
        matrice, noduri = incarca_date_tsp(set_ales)
        self.map_view.draw_test_nodes(noduri)
        self.log("[+] Harta a fost generata si desenata pe canvas.")
        self.plot_panel.clear_plot() # Resetăm graficul vechi
        return matrice

    def _show_plot(self, path: Path) -> None:
        if self.plot_panel.set_plot(path):
            QTimer.singleShot(50, self.plot_panel._apply_pixmap)
            return
        self.log(f"[-] Graficul nu a putut fi afișat: {path}")

    def run_nlp(self) -> None:
        text = self.nlp_input.toPlainText()
        selected_models = [
            check.text() for check in self.nlp_model_checks if check.isChecked()
        ]
        if not selected_models:
            self.nlp_output.setPlainText("Selectați cel puțin un model.")
            return

        self.log("[*] Antrenare clasificator text...")
        try:
            report = self.text_classifier.train(
                selected_models=selected_models,
                ngram_choice=self.nlp_ngram.currentText(),
                max_features=self.nlp_max_features.value(),
                sublinear_tf=self.nlp_sublinear_tf.isChecked(),
                stop_words=self.nlp_stop_words.currentText(),
                domain_dictionary=self.nlp_domain_dict.currentText(),
            )
            prediction = self.text_classifier.classify(
                text,
                domain_dictionary=self.nlp_domain_dict.currentText(),
            )
            steps = preprocess_steps(text, self.nlp_stop_words.currentText())
            sentiment = format_nlp_result(
                analyze_sentiment(text, self.nlp_sentiment_dict.currentText())
            )
            keywords = format_nlp_result(
                extract_keywords(text, stopwords_name=self.nlp_keyword_dict.currentText())
            )
            domain_terms = format_nlp_result(
                detect_domain_terms(text, self.nlp_domain_dict.currentText())
            )
            dictionaries = (
                "\n\nAnaliză pe dicționare selectate:\n\n"
                f"Dicționar de domeniu folosit și ca feature ML: "
                f"{self.nlp_domain_dict.currentText()}\n\n"
                f"{sentiment}\n\n{keywords}\n\n{domain_terms}"
            )
            playground = (
                "\n\nPași preprocesare:\n"
                f"Lowercase: {steps['Lowercase']}\n"
                f"Fără punctuație: {steps['Fără punctuație']}\n"
                f"Tokenizare: {' | '.join(steps['Tokenizare'])}\n" # type: ignore
                f"Fără stop words: {' | '.join(steps['Fără stop words'])}" # type: ignore
            )
            self.nlp_output.setPlainText(
                f"{report}\n\n{prediction}{dictionaries}{playground}"
            )
            self.log("[+] Clasificator text finalizat.")
        except Exception as exc:
            self.nlp_output.setPlainText(f"Eroare: {exc}")
            self.log(f"[-] Eroare clasificator text: {exc}")

    def choose_nlp_dataset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Alege dataset text",
            str(DEMO_DATASET.parent),
            "Dataset text (*.json *.csv)",
        )
        if path:
            self.load_nlp_dataset(Path(path))

    def load_nlp_dataset(self, path: Path) -> None:
        try:
            summary = self.text_classifier.load(
                path, test_size=self.nlp_test_pct.value() / 100.0
            )
            self.nlp_dataset_label.setText(Path(path).name)
            self.nlp_output.setPlainText(summary)
            self.log(f"[+] Dataset text încărcat: {Path(path).name}")
        except Exception as exc:
            self.nlp_output.setPlainText(f"Eroare la încărcare dataset: {exc}")
            self.log(f"[-] Dataset text invalid: {exc}")

    def run_algorithm(self):
        alg_ales = self.algo_combo.currentText()
        set_ales = self.dataset_combo.currentText()
        matrice = self.load_data()

        self.log(f"[*] Se executa {alg_ales}...")
        params = self._collect_algo_params()
        if params:
            self.log(f"[*] Parametri: {params}")

        result = executa_tsp(alg_ales, matrice, algo_params=params)

        self.log(
            f"[+] Executie finalizata. Cost optim: {result.cost:.2f} "
            f"({result.elapsed_sec:.2f}s)"
        )
        self.map_view.draw_path(result.tour)

        run_params = dict(params)
        run_params.update(result.params or {})
        self.benchmark_panel.record_run(
            alg_ales,
            set_ales,
            result.cost,
            result.elapsed_sec,
            run_params,
        )
        self.log(
            f"[+] Benchmark: rulare #{self.benchmark_panel.session.runs[-1].run_id} "
            f"(cost={result.cost:.2f}, timp={result.elapsed_sec:.3f}s)"
        )

        try:
            plot_path = save_performance_plot_or_fallback(result, alg_ales, set_ales)
            self.log(f"[+] Grafic salvat: {plot_path.resolve()}")
            self._show_plot(plot_path)
        except Exception as exc:
            self.log(f"[-] Eroare la generarea graficului: {exc}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = AISolver()
    window.show()
    sys.exit(app.exec())
