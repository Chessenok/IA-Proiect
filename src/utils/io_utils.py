def citeste_matrice(cale_fisier: str) -> list[list[int]]:
    """
    Reads a distance matrix from a text file.

    Format:
        First line: n (number of cities)
        Next n lines: n integers per line

    Args:
        cale_fisier (str): Path to input file.

    Returns:
        list[list[int]]: Distance matrix.
    """

    with open(cale_fisier, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    n = int(lines[0])
    return [
        [int(x) for x in lines[i + 1].split()]
        for i in range(n)
    ]