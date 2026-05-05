import random
from simanneal import Annealer
from tsp_utils import tour_cost


class TSPSolver(Annealer):
    """
    Subclasă a Annealer din simanneal.
    Starea (self.state) este un tur - o listă de indici ai orașelor.
    """

    def __init__(self, state: list[int], distance_matrix: list[list[float]]):
        self.distance_matrix = distance_matrix
        super().__init__(state)

    def move(self):
        """Operator de vecinătate: 2-opt swap."""
        n = len(self.state)
        i, j = sorted(random.sample(range(n), 2))
        self.state[i:j + 1] = self.state[i:j + 1][::-1]

    def energy(self) -> float:
        """Funcția de cost - lungimea totală a turului."""
        return tour_cost(self.state, self.distance_matrix)
