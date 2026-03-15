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
    traseu_initial = [0]

    _backtracking(
        matrice=matrice,
        n=n,
        oras_curent=0,
        vizitat=vizitat,
        traseu_curent=traseu_initial,
        cost_curent=0,
        cea_mai_buna_solutie=cea_mai_buna_solutie
    )

    return cea_mai_buna_solutie['traseu'], cea_mai_buna_solutie['cost']