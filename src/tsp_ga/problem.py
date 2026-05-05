import numpy as np

# ══════════════════════════════════════════════════════════════════
# 1. DEFINIREA PROBLEMEI - ORAȘE ȘI DISTANȚE
# ══════════════════════════════════════════════════════════════════

# Coordonate aproximative (x, y) în km față de un punct de referință
ORASE = {
    0: ("Cluj-Napoca",  (0,    0   )),
    1: ("Brasov",       (220, -130 )),
    2: ("Bucuresti",    (330, -175 )),
    3: ("Timisoara",    (-175, -75 )),
    4: ("Iasi",         (380,   55 )),
    5: ("Constanta",    (450, -225 )),
    6: ("Craiova",      (160, -230 )),
    7: ("Galati",       (430,  -55 )),
    8: ("Oradea",       (-95,   45 )),
    9: ("Sibiu",        (95,   -95 )),
}

N_ORASE = len(ORASE)
COORD = np.array([ORASE[i][1] for i in range(N_ORASE)], dtype=float)
NUME_ORASE = [ORASE[i][0] for i in range(N_ORASE)]


def calculeaza_matrice_distante(coord):
    """Calculează matricea de distanțe euclidiene între toate perechile de orașe."""
    n = len(coord)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dx = coord[i][0] - coord[j][0]
            dy = coord[i][1] - coord[j][1]
            dist[i][j] = np.sqrt(dx**2 + dy**2)
    return dist


DIST_MATRIX = calculeaza_matrice_distante(COORD)


def distanta_ruta(solutie):
    """Calculează distanța totală a unei rute (ciclu complet, revenire la start)."""
    total = 0.0
    n = len(solutie)
    for i in range(n):
        total += DIST_MATRIX[int(solutie[i])][int(solutie[(i + 1) % n])]
    return total