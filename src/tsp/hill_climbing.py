from simpleai.search import (
    SearchProblem,
    hill_climbing_random_restarts
)


class TSPHillClimbing(SearchProblem):
    """
    Travelling Salesman Problem (TSP) formulated as a hill climbing search problem.

    State representation:
        A tour is represented as a list of city indices, e.g. [0, 2, 1, 3].

    Objective:
        Minimize total tour cost (including return to start).

    This class is compatible with SimpleAI search algorithms.
    """

    def __init__(self, matrice: list[list[int]]):
        """
        Initialize the TSP problem.

        Args:
            matrice (list[list[int]]): Distance matrix NxN.
        """
        self.matrice = matrice
        self.n = len(matrice)

        super().__init__(initial_state=list(range(self.n)))

    def actions(self, state):
        """
        Generate possible actions from current state.

        Each action is a swap of two cities in the tour.

        Args:
            state (list[int]): Current tour.

        Returns:
            list[tuple[int, int]]: List of swaps (i, j).
        """
        actions = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                actions.append((i, j))
        return actions

    def result(self, state, action):
        """
        Apply an action (swap two cities).

        Args:
            state (list[int]): Current tour.
            action (tuple[int, int]): Indices to swap.

        Returns:
            list[int]: New tour after swap.
        """
        i, j = action
        new_state = state.copy()
        new_state[i], new_state[j] = new_state[j], new_state[i]
        return new_state

    def value(self, state):
        """
        Compute quality of a state (negative cost for maximization).

        SimpleAI maximizes value, so we return negative distance.

        Args:
            state (list[int]): Tour.

        Returns:
            float: Negative total cost.
        """
        cost = 0
        for i in range(self.n - 1):
            cost += self.matrice[state[i]][state[i + 1]]

        # return to start
        cost += self.matrice[state[-1]][state[0]]

        return -cost

def rezolva_tsp_hc(n: int, matrice: list[list[int]], reporniri: int = 10):
    """
    Solve TSP using Hill Climbing with Random Restarts.

    This function runs multiple hill climbing searches and returns
    the best solution found.

    Args:
        n (int): Number of cities.
        matrice (list[list[int]]): Distance matrix NxN.
        reporniri (int): Number of random restarts.

    Returns:
        tuple:
            best_path (list[int]): Best tour found.
            best_cost (int): Cost of the best tour.
    """

    problem = TSPHillClimbing(matrice)

    result = hill_climbing_random_restarts(
        problem,
        restarts_limit=reporniri
    )

    best_path = result.state # type: ignore

    # compute cost
    cost = 0
    for i in range(n - 1):
        cost += matrice[best_path[i]][best_path[i + 1]]
    cost += matrice[best_path[-1]][best_path[0]]

    return best_path, cost