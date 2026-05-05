import pygad
import time
import random
import numpy as np

from problem import N_ORASE, ORASE
from operators import fitness_func, ox_crossover, swap_mutation, genereaza_populatie
from visualization import plot_convergenta, plot_ruta


# ══════════════════════════════════════════════════════════════════
# 5. RULAREA ALGORITMULUI GENETIC
# ══════════════════════════════════════════════════════════════════

def ruleaza_ga(pop_size=100, n_generatii=500, rata_mutatie=50,
               tip_selectie="tournament", k_tournament=3,
               keep_elitism=2, verbose=True):
    """
    Configurează și rulează GA pentru TSP cu parametrii specificați.
    Returnează: (instanța GA, distanța celei mai bune soluții, durata în secunde)
    """
    populatie_initiala = genereaza_populatie(pop_size, N_ORASE)

    ga_instance = pygad.GA(
        num_generations=n_generatii,
        num_parents_mating=max(2, pop_size // 2),
        fitness_func=fitness_func,
        initial_population=populatie_initiala,
        crossover_type=ox_crossover,
        mutation_type=swap_mutation,
        mutation_percent_genes=rata_mutatie,
        parent_selection_type=tip_selectie,
        K_tournament=k_tournament,
        keep_elitism=keep_elitism,
        keep_parents=0,
        suppress_warnings=True,
    )

    start = time.time()
    ga_instance.run()
    durata = time.time() - start

    solutie, fitness, _ = ga_instance.best_solution()
    distanta = -fitness

    if verbose:
        print(f"\n{'='*55}")
        print(f"Configurație: pop={pop_size}, gen={n_generatii}, mut={rata_mutatie}%")
        ruta_str = " → ".join(ORASE[int(c)][0] for c in solutie)
        print(f"Ruta: {ruta_str} → {ORASE[int(solutie[0])][0]}")
        print(f"Distanță totală: {distanta:.2f}")
        print(f"Timp execuție:   {durata:.2f}s")

    return ga_instance, distanta, durata


# ══════════════════════════════════════════════════════════════════
# 8. PUNCT DE INTRARE
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Importul se face aici pentru a evita dependențele circulare cu funcția ruleaza_ga
    from experiments import studiu_parametri_populatie, studiu_parametri_mutatie, studiu_tip_selectie

    random.seed(42)
    np.random.seed(42)

    print("=== Rulare de bază (pop=100, gen=500, mut=50%) ===")
    ga, dist, durata = ruleaza_ga(pop_size=100, n_generatii=500, rata_mutatie=50)
    solutie, _, _ = ga.best_solution()

    plot_convergenta(ga, titlu="Curba de convergență - configurație de bază")
    plot_ruta(solutie, titlu="Ruta găsită de algoritmul genetic")

    print("\n=== Studiu: impact mărime populație ===")
    studiu_parametri_populatie()

    print("\n=== Studiu: impact rată mutație ===")
    studiu_parametri_mutatie()

    print("\n=== Studiu: tip selecție părinți ===")
    studiu_tip_selectie()