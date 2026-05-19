import random
from typing import List, Tuple

def init_population(pop_size, gene_count):
    return [random.sample(range(gene_count), gene_count)
            for _ in range(pop_size)]

def tournament_selection(pop, fitness_fn, k=3):
    candidates = random.sample(pop, k)
    return max(candidates, key=fitness_fn)

def order_crossover(p1, p2):
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))

    child = [-1] * n
    child[a:b+1] = p1[a:b+1]

    used = set(child[a:b+1])
    fill = [x for x in p2 if x not in used]

    idx = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill[idx]
            idx += 1

    return child

def swap_mutation(sol, mutation_rate):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(sol)), 2)
        sol[i], sol[j] = sol[j], sol[i]
    return sol


def genetic_algorithm(
    fitness_fn,
    gene_count,
    pop_size=50,
    generations=200,
    mutation_rate=0.2,
    selection_k=3
):
    pop = init_population(pop_size, gene_count)

    best = None
    best_fit = float("-inf")

    for _ in range(generations):
        new_pop = []

        for _ in range(pop_size):
            p1 = tournament_selection(pop, fitness_fn, selection_k)
            p2 = tournament_selection(pop, fitness_fn, selection_k)

            child = order_crossover(p1, p2)
            child = swap_mutation(child, mutation_rate)

            new_pop.append(child)

        pop = new_pop

        for ind in pop:
            f = fitness_fn(ind)
            if f > best_fit:
                best_fit = f
                best = ind

    return best, best_fit

def rezolva_tsp_ga(
    matrice: List[List[int]], 
    pop_size: int = 50, 
    generations: int = 200, 
    mutation_rate: float = 0.2, 
    selection_k: int = 3
) -> Tuple[List[int], int]:
    """
    Rezolva TSP utilizand algoritmul genetic, folosind matricea de distante.
    """
    n = len(matrice)
    
    if n == 0:
        return [], 0

    def fitness_fn(tour: List[int]) -> float:
        cost = 0
        for i in range(n - 1):
            cost += matrice[tour[i]][tour[i + 1]]
        cost += matrice[tour[-1]][tour[0]] 
        
        if cost == 0:
            return float('inf')
            
        return 1.0 / cost

    # Rulam nucleul genetic
    best_tour, best_fitness = genetic_algorithm(
        fitness_fn=fitness_fn,
        gene_count=n,
        pop_size=pop_size,
        generations=generations,
        mutation_rate=mutation_rate,
        selection_k=selection_k
    )

    best_cost = 0
    if best_tour:
        for i in range(n - 1):
            best_cost += matrice[best_tour[i]][best_tour[i + 1]]
        best_cost += matrice[best_tour[-1]][best_tour[0]]

    return best_tour, best_cost # type: ignore