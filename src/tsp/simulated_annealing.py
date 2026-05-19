from simanneal import Annealer
from typing import List, Tuple
import random

class TSPSolver(Annealer):
    def __init__(self, state: list[int], distance_matrix):
        self.distance_matrix = distance_matrix
        super().__init__(state)

    def move(self):
        i, j = sorted(random.sample(range(len(self.state)), 2))
        self.state[i:j+1] = self.state[i:j+1][::-1]

    def energy(self):
        return sum(
            self.distance_matrix[self.state[i]][self.state[i+1]]
            for i in range(len(self.state) - 1)
        ) + self.distance_matrix[self.state[-1]][self.state[0]]

def rezolva_tsp_sa(n: int, matrice, init_tour: list[int]):
    solver = TSPSolver(init_tour[:], matrice)

    solver.Tmax = 10000
    solver.Tmin = 1
    solver.steps = 50000

    best_tour, best_cost = solver.anneal()

    return best_tour, best_cost