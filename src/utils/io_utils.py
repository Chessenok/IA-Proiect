import numpy as np
import random


def citeste_matrice(cale):
    """
    Citește o matrice din format text și o returnează ca listă de liste (integers).
    """
    with open(cale, "r") as f:
        linii = f.readlines()
        matrice = []
        for linie in linii:
            # Curățăm spațiile și transformăm fiecare element în int
            rand = [int(x) for x in linie.strip().split()]
            if rand:  # Evităm liniile goale
                matrice.append(rand)
        return matrice


def genereaza_matrice_aleatorie(n, seed=42):
    """
    Generează o matrice de adiacență simetrică pentru TSP.
    """
    random.seed(seed)
    # Inițializăm matricea cu 0
    matrice = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            distanta = random.randint(10, 100)
            matrice[i][j] = distanta
            matrice[j][i] = distanta  # TSP simetric

    return matrice


# Păstrăm și varianta ta cu Numpy dacă ai nevoie de ea în alte părți ale proiectului
def citeste_matrice_numpy_float(cale):
    return np.loadtxt(cale)