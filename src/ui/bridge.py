from pathlib import Path

from tsp.backtracking import rezolva_tsp_backtracking
from tsp.genetic import rezolva_tsp_ga
from tsp.hill_climbing import rezolva_tsp_hc
from tsp.nearest_neighbor import rezolva_tsp_nn
from tsp.simulated_annealing import rezolva_tsp_sa

def obtine_lista_dataseturi(director: str = "data") -> list[str]:
    cale = Path(director).resolve()
    print(f"[*] Caut in directorul: {cale}") # Debug
    return [f.name for f in cale.glob("*.json")]

def executa_tsp(algoritm_nume: str, matrice: list) -> tuple:
    """Interfață unică pentru toți algoritmii TSP."""
    
    # 1. Backtracking
    if "BKT" in algoritm_nume:
        # Implicit ALL mode pentru rezultate optime
        return rezolva_tsp_backtracking(matrice)
    
    # 2. Nearest Neighbor
    elif "NN" in algoritm_nume:
        n = len(matrice)
        return rezolva_tsp_nn(n, matrice)
    
    # 3. Hill Climbing
    elif "HC" in algoritm_nume:
        n = len(matrice)
        return rezolva_tsp_hc(n, matrice)
    
    # 4. Simulated Annealing
    elif "SA" in algoritm_nume:
        n = len(matrice)
        init_tour = list(range(n))
        return rezolva_tsp_sa(n, matrice, init_tour)
    
    # 5. Genetic Algorithm
    elif "GA" in algoritm_nume:
        return rezolva_tsp_ga(matrice)
    
    return [], 0

def executa_nlp(task_nume: str, text: str) -> str:
    return f"Rezultat NLP pentru {task_nume}"