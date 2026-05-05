from tsp_utils import *
from visualization import plot_tour
from annealer import TSPSolver
from cooling import DEFAULT_CONFIG as config
if __name__ == "__main__":
    N = 20                       # număr de orașe
    cities = generate_cities(N)
    dist = build_distance_matrix(cities)

    # Stare inițială: Nearest Neighbour
    init_tour = nearest_neighbour_tour(dist)
    print(f"Cost tur inițial (NN):  {tour_cost(init_tour, dist):.2f}")

    # Configurare SA
    solver = TSPSolver(init_tour[:], dist)   # copie a stării inițiale
    solver.Tmax = config['t_max']    # temperatura maximă
    solver.Tmin = config['t_min']        # temperatura minimă
    solver.steps = config['steps']     # număr total de pași
    solver.updates = config['updates']     # numărul de actualizări afișate în consolă

    # Rulare SA
    best_tour, best_cost = solver.anneal()

    print(f"Cost tur optim (SA):    {best_cost:.2f}")
    print(f"Tur: {best_tour}")

    # Vizualizare
    plot_tour(cities, init_tour, "Tur Initial NN")
    plot_tour(cities, best_tour, "Tur Optim SA simanneal")