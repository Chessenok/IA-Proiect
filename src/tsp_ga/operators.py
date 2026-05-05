import numpy as np
import random
from problem import distanta_ruta

# ══════════════════════════════════════════════════════════════════
# 2. FUNCȚIA DE FITNESS
# ══════════════════════════════════════════════════════════════════

def fitness_func(ga_instance, solutie, solutie_idx):
    """PyGAD maximizează fitness-ul → returnăm negativul distanței."""
    return -distanta_ruta(solutie)


# ══════════════════════════════════════════════════════════════════
# 3. OPERATORI PERSONALIZAȚI - CROSSOVER ȘI MUTAȚIE
# ══════════════════════════════════════════════════════════════════

def ox_crossover(parinti, offspring_size, ga_instance):
    """
    Order Crossover (OX): produce mereu permutări valide din doi părinți.

    Algoritmul:
    1. Copiază segmentul [cx1..cx2] din parent1 în copil.
    2. Completează pozițiile libere cu genele din parent2 (în ordinea lor),
       omițând genele deja prezente în copil.
    """
    offspring = []
    idx = 0
    while len(offspring) < offspring_size[0]:
        p1 = parinti[idx % parinti.shape[0]].astype(int).tolist()
        p2 = parinti[(idx + 1) % parinti.shape[0]].astype(int).tolist()
        n = len(p1)

        cx1, cx2 = sorted(random.sample(range(n), 2))

        copil = [-1] * n
        copil[cx1:cx2 + 1] = p1[cx1:cx2 + 1]

        set_segment = set(copil[cx1:cx2 + 1])
        gene_ramase = [g for g in p2 if g not in set_segment]
        pozitii_libere = [i for i in range(n) if copil[i] == -1]

        for pos, gena in zip(pozitii_libere, gene_ramase):
            copil[pos] = gena

        offspring.append(copil)
        idx += 1

    return np.array(offspring, dtype=int)


def swap_mutation(offspring, ga_instance):
    """
    Swap Mutation: cu probabilitate rata_mutatie, schimbă două gene aleatoare.
    Constrângerea de permutare este menținută prin construcție.
    """
    rata = ga_instance.mutation_percent_genes / 100.0
    for i in range(offspring.shape[0]):
        if random.random() < rata:
            n = offspring.shape[1]
            idx1, idx2 = random.sample(range(n), 2)
            temp = int(offspring[i][idx1])
            offspring[i][idx1] = offspring[i][idx2]
            offspring[i][idx2] = temp
    return offspring


# ══════════════════════════════════════════════════════════════════
# 4. GENERARE POPULAȚIE INIȚIALĂ
# ══════════════════════════════════════════════════════════════════

def genereaza_populatie(pop_size, n_orase):
    """Generează pop_size permutări aleatoare distincte ale celor n_orase orașe."""
    pop = []
    for _ in range(pop_size):
        perm = list(range(n_orase))
        random.shuffle(perm)
        pop.append(perm)
    return np.array(pop, dtype=int)