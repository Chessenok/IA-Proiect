import time
from sys import maxsize
from enum import Enum
from typing import List, Tuple, Optional

class StopMode(Enum):
    ALL = "all"
    FIRST = "first"
    TIME = "time"
    CALLS = "calls"

def rezolva_tsp_backtracking(
    matrice: List[List[int]],
    mode: StopMode = StopMode.ALL,
    time_limit: Optional[float] = None,
    max_calls: Optional[int] = None
) -> Tuple[List[int], int]:
    
    n = len(matrice)
    best_cost = maxsize
    best_path: List[int] = []

    # Tratarea cazului de harta goala
    if n == 0:
        return [], 0

    start_time = time.perf_counter()
    node_visits = 0
    stop = False

    def time_exceeded() -> bool:
        return (
            mode == StopMode.TIME
            and time_limit is not None
            and (time.perf_counter() - start_time > time_limit)
        )

    def calls_exceeded() -> bool:
        return (
            mode == StopMode.CALLS
            and max_calls is not None
            and node_visits >= max_calls
        )
    
    def backtrack(current: int, visited: List[bool], path: List[int], cost: int):
        nonlocal best_cost, best_path, node_visits, stop

        if stop:
            return

        if time_exceeded() or calls_exceeded():
            stop = True
            return

        node_visits += 1

        # Pruning (optimizare): tăiem ramura dacă deja costul e mai mare decât cel mai bun găsit
        if mode != StopMode.FIRST and cost >= best_cost:
            return

        # Soluție completă găsită
        if len(path) == n:
            total_cost = cost + matrice[current][path[0]]

            if total_cost < best_cost:
                best_cost = total_cost
                best_path = path.copy()

            if mode == StopMode.FIRST:
                stop = True

            return

        # Explorarea vecinilor
        for nxt in range(n):
            if visited[nxt]:
                continue

            visited[nxt] = True
            path.append(nxt)

            backtrack(
                nxt,
                visited,
                path,
                cost + matrice[current][nxt]
            )

            # Backtrack (anularea pasului)
            path.pop()
            visited[nxt] = False

    visited = [False] * n
    visited[0] = True
    
    backtrack(0, visited, [0], 0)

    return best_path, best_cost