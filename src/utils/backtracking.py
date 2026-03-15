import sys
import time

import numpy as np
import io_utils as io
def _backtracking(matrice, n, oras_curent, vizitat, traseu, cost):
    """Explorare recursiva a spatiului de solutii TSP prin backtracking.

    La fiecare apel recursiv se incearca extinderea traseului curent cu un
    oras nevizitat. Ramurile al caror cost partial depaseste minimul global
    cunoscut sunt abandonate imediat (prunere branch-and-bound).

    Args:
        matrice: Matricea de distante NxN (lista de liste de intregi).
        n: Numarul de orase (int).
        oras_curent: Indexul orasului in care ne aflam la pasul curent (int).
        vizitat: Lista de booleeni de lungime n; vizitat[i] este True daca
            orasul i a fost deja inclus in traseu.
        traseu: Lista cu orasele vizitate pana acum, in ordinea parcurgerii.
            Primul element este intotdeauna 0 (orasul de start).
        cost: Costul acumulat al traseului partial curent (int sau float).
    """
    global _cost_minim, _traseu_optim

    # Caz de baza: toate orasele au fost vizitate — inchidem turul.
    if len(traseu) == n:
        cost_total = cost + matrice[oras_curent][traseu[0]]
        if cost_total < _cost_minim:
            _cost_minim = cost_total
            _traseu_optim = traseu[:]  # copie a listei curente
        return

    # Pas recursiv: incercam extinderea traseului cu fiecare oras nevizitat.
    for urmator in range(n):
        if vizitat[urmator]:
            continue

        cost_nou = cost + matrice[oras_curent][urmator]

        # Prunere: abandonam ramura daca costul partial nu poate imbunatati
        # solutia optima cunoscuta (toate distantele sunt strict pozitive).
        if cost_nou >= _cost_minim:
            continue

        vizitat[urmator] = True
        traseu.append(urmator)

        _backtracking(matrice, n, urmator, vizitat, traseu, cost_nou)

        # Revenire (backtrack): restauram starea pentru a explora alte ramuri.
        traseu.pop()
        vizitat[urmator] = False


def rezolva_tsp(cale_fisier):
    """Rezolva TSP prin backtracking recursiv cu prunere branch-and-bound.

    Citeste datele din fisierul specificat, ruleaza algoritmul de backtracking
    si afiseaza traseul optim, costul minim si timpul de executie.

    Args:
        cale_fisier: Calea catre fisierul text cu matricea de distante (str).
    """
    global _cost_minim, _traseu_optim

    matrice = io.citesteMatriceInt(cale_fisier)
    n =  len(matrice)
    print(f"Numar de orase: {n}")
    print("Matricea de distante:")
    for rand in matrice:
        print("  " + "  ".join(f"{val:2d}" for val in rand))
    print()

    # Resetam variabilele globale pentru a permite apeluri repetate.
    _cost_minim = sys.maxsize
    _traseu_optim = []

    # Fixam orasul de start la indexul 0 (optimizare pentru TSP simetric:
    # elimina N rotatii echivalente ale aceluiasi tur).
    vizitat = [False] * n
    vizitat[0] = True

    start = time.perf_counter()
    _backtracking(matrice, n, 0, vizitat, [0], 0)
    durata = time.perf_counter() - start

    if _traseu_optim:
        sir_traseu = " -> ".join(map(str, _traseu_optim))
        sir_traseu += f" -> {_traseu_optim[0]}"
        print(f"Traseu optim:   {sir_traseu}")
        print(f"Cost minim:     {_cost_minim}")
    else:
        print("Nu a fost gasit niciun traseu valid.")

    print(f"Timp de executie: {durata:.6f} secunde")


