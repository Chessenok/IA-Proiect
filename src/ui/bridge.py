import time
from pathlib import Path

from tsp.backtracking import rezolva_tsp_backtracking
from tsp.genetic import rezolva_tsp_ga
from tsp.hill_climbing import rezolva_tsp_hc
from tsp.nearest_neighbor import rezolva_tsp_nn
from tsp.simulated_annealing import rezolva_tsp_sa
from utils.metrics import RunResult


def obtine_lista_dataseturi(director: str = "data") -> list[str]:
    cale = Path(director).resolve()
    print(f"[*] Caut in directorul: {cale}")
    return [f.name for f in cale.glob("*.json")]


def executa_tsp(
    algoritm_nume: str,
    matrice: list,
    algo_params: dict | None = None,
) -> RunResult:
    """Interfață unică pentru toți algoritmii TSP."""
    params = algo_params or {}
    t0 = time.perf_counter()

    if "BKT" in algoritm_nume:
        traseu, cost = rezolva_tsp_backtracking(matrice)
    elif "NN" in algoritm_nume:
        n = len(matrice)
        traseu, cost = rezolva_tsp_nn(n, matrice)
    elif "HC" in algoritm_nume:
        n = len(matrice)
        traseu, cost = rezolva_tsp_hc(n, matrice)
    elif "SA" in algoritm_nume:
        n = len(matrice)
        init_tour = list(range(n))
        traseu, cost, history, sa_params = rezolva_tsp_sa(
            n,
            matrice,
            init_tour,
            Tmax=float(params.get("Tmax", 10000)),
            Tmin=float(params.get("Tmin", 1)),
            steps=int(params.get("steps", 50000)),
            updates=int(params.get("updates", 100)),
        )
        elapsed = time.perf_counter() - t0
        return RunResult(
            tour=traseu,
            cost=float(cost),
            history=history,
            history_xlabel="Update",
            elapsed_sec=elapsed,
            params=sa_params,
        )
    elif "GA" in algoritm_nume:
        traseu, cost, history, ga_params = rezolva_tsp_ga(
            matrice,
            pop_size=int(params.get("pop_size", 50)),
            generations=int(params.get("generations", 200)),
            mutation_rate=float(params.get("mutation_rate", 0.2)),
            selection_k=int(params.get("selection_k", 3)),
        )
        elapsed = time.perf_counter() - t0
        return RunResult(
            tour=traseu,
            cost=float(cost),
            history=history,
            history_xlabel="Generation",
            elapsed_sec=elapsed,
            params=ga_params,
        )
    else:
        traseu, cost = [], 0

    elapsed = time.perf_counter() - t0
    return RunResult(
        tour=traseu,
        cost=float(cost),
        history=[float(cost)] if cost else [],
        elapsed_sec=elapsed,
        params=dict(params),
    )


def executa_nlp(task_nume: str, text: str) -> str:
    from nlp.classifier import DEMO_DATASET, TextClassifierRunner

    runner = TextClassifierRunner()
    runner.load(DEMO_DATASET)
    report = runner.train(
        selected_models=["Naive Bayes (MultinomialNB)"],
        ngram_choice="Unigrame (1,1)",
        max_features=3000,
        sublinear_tf=True,
        stop_words="romanian",
        domain_dictionary="drone_civil_terms",
    )
    return f"{report}\n\n{runner.classify(text, domain_dictionary='drone_civil_terms')}"
