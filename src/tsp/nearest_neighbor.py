import time
from sys import maxsize
from typing import List, Tuple

def rezolva_tsp_nn(n: int, matrice: List[List[int]], start: int = 0) -> Tuple[List[int], int]:
    """
    Nearest Neighbor heuristic (single start).

    Returns:
        (path, cost)
    """

    visited = [False] * n
    visited[start] = True

    path = [start]
    cost = 0
    current: int = start

    for _ in range(n - 1):
        next_city: int|None = None
        best_dist = maxsize

        for j in range(n):
            if not visited[j] and matrice[current][j] < best_dist:
                best_dist = matrice[current][j]
                next_city = j

        visited[next_city] = True # type: ignore
        path.append(next_city) # type: ignore
        cost += best_dist
        current = next_city # type: ignore

    cost += matrice[current][start]
    path.append(start)

    return path, cost

def rezolva_tsp_nn_multistart(n: int, matrice: List[List[int]]):
    """
    Runs NN from every possible start node.

    Returns:
        (best_path, best_cost)
    """

    best_cost = maxsize
    best_path = []

    for start in range(n):
        path, cost = rezolva_tsp_nn(n, matrice, start)

        if cost < best_cost:
            best_cost = cost
            best_path = path

    return best_path, best_cost

def rezolva_tsp_nn_timp(n: int, matrice: List[List[int]], timp_max: float):
    """
    Nearest Neighbor with time constraint (multi-start until time runs out).
    """

    start_time = time.perf_counter()

    best_cost = maxsize
    best_path = []

    start = 0

    while time.perf_counter() - start_time < timp_max:
        path, cost = rezolva_tsp_nn(n, matrice, start)

        if cost < best_cost:
            best_cost = cost
            best_path = path

        start = (start + 1) % n

    return best_path, best_cost

from aima3.search import Problem, hill_climbing
from typing import List, Tuple


class TSPProblem(Problem):
    """
    TSP formulated as a search problem for AIMA hill climbing.

    State representation:
        A permutation of cities (tour), e.g. [0, 2, 1, 3].

    Objective:
        Minimize total tour cost (implemented as maximization of negative cost).
    """

    def __init__(self, matrice: List[List[int]]):
        self.matrice = matrice
        self.n = len(matrice)

        # initial state: simple ordered tour
        initial_state = list(range(self.n))

        super().__init__(initial_state)

    def value(self, state: List[int]) -> float:
        """
        Computes fitness of a state (negative tour cost).

        AIMA hill_climbing maximizes value, so we return -cost.
        """
        cost = 0

        for i in range(self.n - 1):
            cost += self.matrice[state[i]][state[i + 1]]

        # return to start
        cost += self.matrice[state[-1]][state[0]]

        return -cost


def rezolva_tsp_nn_aima(
    n: int,
    matrice: List[List[int]],
    start: int = 0
) -> Tuple[List[int], int]:
    """
    Solves TSP using AIMA hill climbing (NN-like heuristic approximation).

    Args:
        n: number of cities
        matrice: NxN distance matrix
        start: starting city (ignored by AIMA initial permutation logic, kept for compatibility)

    Returns:
        (best_path, best_cost)
    """

    problem = TSPProblem(matrice)
    result = hill_climbing(problem)

    path = result.state

    cost = 0
    for i in range(n - 1):
        cost += matrice[path[i]][path[i + 1]]
    cost += matrice[path[-1]][path[0]]

    return path, cost


def rezolva_tsp_nn_aima_multistart(
    n: int,
    matrice: List[List[int]]
) -> Tuple[List[int], int]:
    """
    Runs AIMA hill climbing multiple times (random restarts behavior).

    Returns:
        (best_path, best_cost)
    """

    best_cost = float("inf")
    best_path = []

    for _ in range(n):
        path, cost = rezolva_tsp_nn_aima(n, matrice)

        if cost < best_cost:
            best_cost = cost
            best_path = path

    return best_path, best_cost # type: ignore