from simanneal import Annealer
import random


class TSPSolver(Annealer):
    def __init__(self, state: list[int], distance_matrix):
        self.distance_matrix = distance_matrix
        self.history: list[float] = []
        super().__init__(state)

    def move(self):
        i, j = sorted(random.sample(range(len(self.state)), 2))
        self.state[i : j + 1] = self.state[i : j + 1][::-1]

    def energy(self):
        return sum(
            self.distance_matrix[self.state[i]][self.state[i + 1]]
            for i in range(len(self.state) - 1)
        ) + self.distance_matrix[self.state[-1]][self.state[0]]

    def update(self, step, T, E, acceptance, improvement):
        self.history.append(float(self.best_energy))
        self.default_update(step, T, E, acceptance, improvement)


def rezolva_tsp_sa(
    n: int,
    matrice,
    init_tour: list[int],
    *,
    Tmax: float = 10000.0,
    Tmin: float = 1.0,
    steps: int = 50000,
    updates: int = 100,
) -> tuple[list[int], int, list[float], dict]:
    solver = TSPSolver(init_tour[:], matrice)

    solver.Tmax = Tmax
    solver.Tmin = Tmin
    solver.steps = steps
    solver.updates = updates

    solver.history.append(float(solver.energy()))
    best_tour, best_cost = solver.anneal()

    params = {
        "Tmax": Tmax,
        "Tmin": Tmin,
        "steps": steps,
        "updates": updates,
    }
    return best_tour, int(best_cost), solver.history, params
