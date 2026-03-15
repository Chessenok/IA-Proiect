import os
import random
import time
import multiprocessing
import matplotlib.pyplot as plt

# Importăm modulele tale
import backtracking
import hill_climbing_tsp as hctsp

def genereaza_matrice_simetrica(n, seed=42):
    """Generează o matrice de distanțe simetrică cu valori între 1 și 100."""
    random.seed(seed)
    matrice = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            valoare = random.randint(1, 100)
            matrice[i][j] = valoare
            matrice[j][i] = valoare
    return matrice

def salveaza_matrice_temp(matrice, fisier_temp="temp_matrix.txt"):
    """Salvează matricea pe disc pentru a putea fi citită de backtracking."""
    with open(fisier_temp, "w") as f:
        for rand in matrice:
            f.write(" ".join(map(str, rand)) + "\n")

def wrapper_backtracking(fisier):
    """O funcție simplă care doar apelează backtracking-ul colegului."""
    # Am ascuns print-urile interne ale backtracking-ului pentru a nu polua terminalul
    # Dacă vrei să vezi printurile colegului, poți scoate redirectionarea
    import sys
    sys.stdout = open(os.devnull, 'w')
    backtracking.rezolva_tsp(fisier)
    sys.stdout.close()

def ruleaza_bt_cu_timeout(fisier, timeout=30):
    """Rulează backtracking-ul într-un proces separat, forțând oprirea la timeout."""
    start = time.perf_counter()
    p = multiprocessing.Process(target=wrapper_backtracking, args=(fisier,))
    p.start()
    p.join(timeout)
    
    if p.is_alive():
        # A depășit limita de timp!
        p.terminate()
        p.join()
        return None # Returnăm None pentru a semnala timeout-ul
        
    durata = time.perf_counter() - start
    return durata

def ruleaza_experiment():
    """Rulează experimentul comparativ și generează graficul de performanță."""
    valori_n_bt = [5, 7, 8, 10, 12]
    valori_n_hc = [5, 7, 8, 10, 12, 15, 20, 30, 50]
    
    timpi_bt = []
    timpi_hc = []
    
    fisier_temp = "temp_tsp_experiment.txt"
    seed_fix = 42

    print("=== START EXPERIMENT PERFORMANTA ===")
    
    # --- Rulăm pentru Backtracking ---
    print("\n1. Testare Backtracking (limita 30 secunde)...")
    for n in valori_n_bt:
        matrice = genereaza_matrice_simetrica(n, seed=seed_fix)
        salveaza_matrice_temp(matrice, fisier_temp)
        
        print(f"  -> Rulare BT pentru N={n}...", end="", flush=True)
        timp = ruleaza_bt_cu_timeout(fisier_temp, timeout=30)
        
        if timp is None:
            print(" TIMEOUT! (>30 secunde)")
            timpi_bt.append(None)
        else:
            print(f" {timp:.4f} secunde.")
            timpi_bt.append(timp)

    # --- Rulăm pentru Hill Climbing ---
    print("\n2. Testare Hill Climbing...")
    for n in valori_n_hc:
        matrice = genereaza_matrice_simetrica(n, seed=seed_fix)
        # Nu mai salvam pe disc, ii pasam direct matricea (conform discutiei anterioare)
        
        print(f"  -> Rulare HC pentru N={n}...", end="", flush=True)
        start_hc = time.perf_counter()
        # Apelam HC (il ignoram pe traseu_optim si cost_minim, ne intereseaza doar timpul)
        _, _ = hctsp.rezolva_tsp_hc(matrice)
        timp = time.perf_counter() - start_hc
        print(f" {timp:.4f} secunde.")
        timpi_hc.append(timp)

    # Curățăm fișierul temporar
    if os.path.exists(fisier_temp):
        os.remove(fisier_temp)

    print("\nExperiment finalizat! Generare grafice...")
    genereaza_grafice(valori_n_bt, timpi_bt, valori_n_hc, timpi_hc)


def genereaza_grafice(n_bt, timpi_bt, n_hc, timpi_hc):
    """Desenează cele două subploturi cerute."""
    # Curățăm lista de timpi Backtracking de valorile None (dacă au luat timeout)
    # ca să putem trasa graficul corect
    n_bt_valid = [n for n, t in zip(n_bt, timpi_bt) if t is not None]
    timpi_bt_valid = [t for t in timpi_bt if t is not None]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Comparație Performanță: Backtracking vs Hill Climbing Random Restarts', fontsize=14)

    # --- Subplot 1: Scară liniară ---
    ax1.plot(n_hc, timpi_hc, marker='o', color='blue', label='Hill Climbing')
    ax1.plot(n_bt_valid, timpi_bt_valid, marker='s', color='red', label='Backtracking')
    ax1.set_title('Timp de execuție (Scară Liniară)')
    ax1.set_xlabel('Număr de orașe (N)')
    ax1.set_ylabel('Timp (secunde)')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    # --- Subplot 2: Scară logaritmică ---
    ax2.semilogy(n_hc, timpi_hc, marker='o', color='blue', label='Hill Climbing')
    ax2.semilogy(n_bt_valid, timpi_bt_valid, marker='s', color='red', label='Backtracking')
    ax2.set_title('Timp de execuție (Scară Logaritmică)')
    ax2.set_xlabel('Număr de orașe (N)')
    ax2.set_ylabel('Timp (secunde - LOG)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Salvare și afișare
    nume_fisier = 'comparare_performanta.png'
    plt.tight_layout()
    plt.savefig(nume_fisier)
    print(f"Graficul a fost salvat cu succes ca '{nume_fisier}' în folderul curent.")
    # plt.show() # Poti decomenta asta daca vrei sa iti si apara fereastra pe ecran

if __name__ == '__main__':
    ruleaza_experiment()