import math
import random


# ─── Generare date ────────────────────────────────────────────────────────────

def generate_cities(n: int, seed: int = 42) -> list[tuple[float, float]]:
    """Generează n orașe cu coordonate aleatoare în [0, 100]."""
    random.seed(seed)
    return [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n)]


def euclidean_distance(c1: tuple, c2: tuple) -> float:
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def build_distance_matrix(cities: list) -> list[list[float]]:
    n = len(cities)
    return [[euclidean_distance(cities[i], cities[j]) for j in range(n)]
            for i in range(n)]


def tour_cost(tour: list, dist: list[list[float]]) -> float:
    return sum(dist[tour[i]][tour[(i + 1) % len(tour)]] for i in range(len(tour)))


# ─── Stare inițială: Nearest Neighbour ────────────────────────────────────────

def nearest_neighbour_tour(dist: list[list[float]], start: int = 0) -> list[int]:
    n = len(dist)
    visited = [False] * n
    tour = [start]
    visited[start] = True
    for _ in range(n - 1):
        last = tour[-1]
        nearest = min((dist[last][j], j) for j in range(n) if not visited[j])[1]
        tour.append(nearest)
        visited[nearest] = True
    return tour