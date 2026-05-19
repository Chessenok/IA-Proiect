import math
import random
import os
import json
from typing import List, Tuple

def genereaza_date_tsp(n: int, latime: int = 800, inaltime: int = 600) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
    """
    Genereaza n coordonate aleatorii si matricea de distante aferenta.
    
    Returns:
        - matrice: Matricea NxN de distante (int)
        - noduri: Lista de tuple (x, y) pentru vizualizare in UI
    """
    # 1. Generam coordonatele nodurilor
    noduri = [(random.randint(50, latime-50), random.randint(50, inaltime-50)) for _ in range(n)]
    
    # 2. Initializam matricea cu 0
    matrice = [[0 for _ in range(n)] for _ in range(n)]
    
    # 3. Calculam distantele euclidiene
    for i in range(n):
        for j in range(i + 1, n):
            dist = math.sqrt((noduri[i][0] - noduri[j][0])**2 + (noduri[i][1] - noduri[j][1])**2)
            matrice[i][j] = int(dist)
            matrice[j][i] = int(dist) # Simetrie
            
    return matrice, noduri

def salveaza_date_tsp(filename: str, matrice: list, noduri: list):
    """
    Salveaza matricea si coordonatele intr-un fisier JSON.
    """
    date_de_salvat = {
        "n": len(matrice),
        "matrice": matrice,
        "noduri": noduri
    }
    
    # Asiguram directorul 'data' exista
    if not os.path.exists('data'):
        os.makedirs('data')
        
    path = os.path.join('data', filename)
    
    with open(path, 'w') as f:
        json.dump(date_de_salvat, f, indent=4)
        
    return path

def incarca_date_tsp(filename: str):
    """
    Incarca matricea si coordonatele dintr-un fisier JSON.
    """
    path = os.path.join('data', filename)
    with open(path, 'r') as f:
        data = json.load(f)
    return data["matrice"], data["noduri"]