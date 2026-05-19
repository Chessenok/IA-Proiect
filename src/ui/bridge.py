# ui/bridge.py
from tsp.backtracking import rezolva_tsp_backtracking
from tsp.genetic import rezolva_tsp_ga
# importă restul algoritmilor aici...

def executa_tsp(algoritm_nume: str, matrice: list) -> tuple:
    """Interfață unică pentru toți algoritmii TSP."""
    if "BKT" in algoritm_nume:
        return rezolva_tsp_backtracking(matrice)
    elif "GA" in algoritm_nume:
        return rezolva_tsp_ga(matrice)
    # ... restul algoritmilor
    return [], 0

def executa_nlp(task_nume: str, text: str) -> str:
    return f"Rezultat NLP pentru {task_nume}"