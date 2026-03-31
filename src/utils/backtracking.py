
import sys

def _backtracking(matrice, n, oras_curent, vizitat, traseu_curent, cost_curent, cea_mai_buna_solutie):
    """
    Explorare recursivă a spațiului de soluții TSP prin backtracking.

    Funcție internă care implementează logica de generare a traseelor și prunere
    (branch-and-bound). Utilizează un dicționar mutabil pentru a reține
    costul minim și traseul optim la nivel global.

    Args:
        matrice (list of list of int): Matricea de distanțe NxN.
        n (int): Numărul total de orașe.
        oras_curent (int): Indexul orașului vizitat la pasul curent.
        vizitat (list of bool): Starea de vizitare a fiecărui oraș.
        traseu_curent (list of int): Orașele vizitate până în prezent.
        cost_curent (int): Costul acumulat al traseului parțial curent.
        cea_mai_buna_solutie (dict): Un dicționar mutabil cu cheile 'traseu' 
            (list of int) și 'cost' (int), care stochează minimul global curent.
    """
    if len(traseu_curent) == n:
        cost_total = cost_curent + matrice[oras_curent][traseu_curent[0]]
        
        if cost_total < cea_mai_buna_solutie['cost']:
            cea_mai_buna_solutie['cost'] = cost_total
            cea_mai_buna_solutie['traseu'] = traseu_curent[:]
        return

    for urmator in range(n):
        if not vizitat[urmator]:
            cost_nou = cost_curent + matrice[oras_curent][urmator]

            if cost_nou < cea_mai_buna_solutie['cost']:
                vizitat[urmator] = True
                traseu_curent.append(urmator)

                _backtracking(
                    matrice, n, urmator, vizitat, traseu_curent, 
                    cost_nou, cea_mai_buna_solutie
                )

                traseu_curent.pop()
                vizitat[urmator] = False


def rezolva_tsp(matrice):
    """
    Rezolvă Problema Comis-Voiajorului (TSP) folosind backtracking optimizat.

    Primește direct matricea de distanțe și inițializează căutarea drumului 
    de cost minim care trece prin toate orașele exact o dată și se întoarce 
    la punctul de plecare.

    Args:
        matrice (list of list of int): Matricea de distanțe NxN.

    Returns:
        tuple: Un tuplu conținând:
            - traseu_optim (list of int): Lista cu indicii orașelor în ordinea
              optimă de parcurgere.
            - cost_minim (int): Costul total al traseului optim. 
            (Returnează ([], 0) dacă nu există soluție sau matricea e prea mică).
    """
    n = len(matrice)
    
    if n <= 1:
        return ([0] if n == 1 else [], 0)

    cea_mai_buna_solutie = {
        'traseu': [],
        'cost': sys.maxsize
    }

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
