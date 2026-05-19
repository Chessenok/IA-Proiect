def manhattan(v1: list[int], v2: list[int]) -> int:
    """
    Compute the Manhattan distance between two integer vectors.

    Args:
        v1 (list[int]): The first vector.
        v2 (list[int]): The second vector.

    Raises:
        ValueError: If the two vectors do not have the same length.

    Returns:
        int: The Manhattan distance between v1 and v2.
    """
    if len(v1) != len(v2):
        raise ValueError("The 2 vectors must be the same size!")

    return sum(abs(x - y) for x, y in zip(v1, v2))
