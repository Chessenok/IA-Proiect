<<<<<<< Updated upstream
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


=======
import time


def rezolva_tsp_backtracking(n, matrice, mod='toate', timp_max=None, y_max=None):
    """
    Implementează algoritmul Backtracking pentru TSP cu 4 moduri de oprire.

    Returnează: (best_path, min_cost, stats)
    """
    start_time = time.time()

    # Variabile de stare pentru algoritm
    best_path = []
    min_cost = float('inf')
    solutii_gasite = 0
    stop_flag = False

    # Calea curentă începe mereu din orașul 0 pentru a evita permutările circulare
    path_curent = [0]
    vizitat = [False] * n
    vizitat[0] = True

    def backtrack(u, cost_curent):
        nonlocal min_cost, best_path, solutii_gasite, stop_flag

        # Verificare limită de timp (modul 'timp')
        if mod == 'timp' and (time.time() - start_time) > timp_max:
            stop_flag = True
            return

        # Condiție de oprire dacă am găsit deja destule soluții (modul 'y_solutii')
        if mod == 'y_solutii' and solutii_gasite >= y_max:
            stop_flag = True
            return

        # Pruning: dacă deja am depășit costul minim, nu are sens să continuăm pe ramura asta
        # (Nu se aplică la modul 'prima' pentru că acolo vrem doar prima soluție validă)
        if mod != 'prima' and cost_curent >= min_cost:
            return

        # Am vizitat toate orașele?
        if len(path_curent) == n:
            # Verificăm dacă există drum înapoi la start
            distanta_retur = matrice[u][0]
            if distanta_retur > 0:  # Presupunem 0 sau -1 pentru lipsă muchie
                total_cost = cost_curent + distanta_retur
                solutii_gasite += 1

                # Actualizăm cea mai bună soluție
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_path = list(path_curent) + [0]

                # Modul 'prima' - ne oprim imediat după prima soluție completă
                if mod == 'prima':
                    stop_flag = True

            return

        # Explorăm vecinii
        for v in range(n):
            if not vizitat[v] and matrice[u][v] > 0:
                vizitat[v] = True
                path_curent.append(v)

                backtrack(v, cost_curent + matrice[u][v])

                # Backtrack (curățare)
                path_curent.pop()
                vizitat[v] = False

                if stop_flag:
                    return

    # Lansăm algoritmul
    backtrack(0, 0)

    # Colectăm metrice de performanță
    stats = {
        "timp_executie": time.time() - start_time,
        "solutii_gasite": solutii_gasite,
        "mod_utilizat": mod
    }

    return best_path, min_cost, stats
>>>>>>> Stashed changes
