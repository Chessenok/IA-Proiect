import time
import utils.io_utils as io
import utils.backtracking as backtracking
import utils.hill_climbing_tsp as hctsp
import utils.performance as performance

def afiseaza_rezultat(nume_algoritm, traseu, cost, durata):
    """Formatează și afișează rezultatul într-un mod clar."""
    print(f"--- {nume_algoritm} ---")
    if not traseu: # Corectat din varianta anterioară
        print("Nu a fost găsit niciun traseu valid.")
    else:
        sir_traseu = " -> ".join(map(str, traseu)) + f" -> {traseu[0]}"
        print(f"Traseu optim:   {sir_traseu}")
        print(f"Cost minim:     {cost}")
    print(f"Timp execuție:  {durata:.6f} secunde\n")

def main():
    # 1. Ajustează calea către fișierul tău de test aici
    cale_fisier = "Test Files/orase7.txt" 
    
    print("=" * 50)
    print(" PARTEA 1: REZOLVARE INSTANȚĂ DE TEST")
    print("=" * 50)
    
    try:
        matrice = io.citesteMatriceInt(cale_fisier)
        print(f"Matrice citită cu succes ({len(matrice)} orașe).\n")
        
        # Rulare Backtracking
        start_bt = time.perf_counter()
        t_bt, c_bt = backtracking.rezolva_tsp(matrice)
        durata_bt = time.perf_counter() - start_bt
        afiseaza_rezultat("Backtracking", t_bt, c_bt, durata_bt)
        
        # Rulare Hill Climbing
        start_hc = time.perf_counter()
        t_hc, c_hc = hctsp.rezolva_tsp_hc(matrice)
        durata_hc = time.perf_counter() - start_hc
        afiseaza_rezultat("Hill Climbing Random Restarts", t_hc, c_hc, durata_hc)
        
    except FileNotFoundError:
        print(f"[!] Fișierul {cale_fisier} nu a fost găsit. Sari peste Partea 1...\n")

    print("=" * 50)
    print(" PARTEA 2: EXPERIMENT DE PERFORMANȚĂ (GRAFICE)")
    print("=" * 50)
    
    # Rulăm experimentul care va dura puțin (datorită limitelor de timeout)
    performance.ruleaza_experiment()
    
    print("\nExecuție completă! Verifică folderul proiectului pentru graficul .png.")

if __name__ == "__main__":
    main()