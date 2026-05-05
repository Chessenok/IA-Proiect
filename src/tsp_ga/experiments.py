import matplotlib.pyplot as plt
from main import ruleaza_ga

# ══════════════════════════════════════════════════════════════════
# 7. STUDIU DE PARAMETRI - EXEMPLU COMPLET
# ══════════════════════════════════════════════════════════════════

def studiu_parametri_populatie():
    """
    Sarcina 2: Compară configurații cu mărimi de populație diferite.
    Generează curbe de convergență suprapuse și grafic comparativ.
    """
    valori_pop = [20, 50, 100, 200]
    culori = ['tomato', 'steelblue', 'seagreen', 'darkorange']

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    distante_finale = []
    durate = []

    for pop, culoare in zip(valori_pop, culori):
        ga, dist, durata = ruleaza_ga(
            pop_size=pop, n_generatii=300, rata_mutatie=40, verbose=False
        )
        distante_finale.append(dist)
        durate.append(durata)

        d = [-f for f in ga.best_solutions_fitness]
        axes[0].plot(d, color=culoare, linewidth=1.5, label=f"pop={pop}")

    axes[0].set_xlabel("Generație")
    axes[0].set_ylabel("Distanță totală")
    axes[0].set_title("Curbe de convergență - mărimi de populație diferite")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].bar([str(p) for p in valori_pop], distante_finale,
                color=culori, alpha=0.85)
    axes[1].set_xlabel("Mărimea populației (sol_per_pop)")
    axes[1].set_ylabel("Distanță finală (mai mică = mai bun)")
    axes[1].set_title("Distanță finală în funcție de mărimea populației")
    for j, (v, d) in enumerate(zip(valori_pop, distante_finale)):
        axes[1].text(j, d + 2, f"{d:.1f}", ha='center', va='bottom', fontsize=9)

    axes[2].bar([str(p) for p in valori_pop], durate, color=culori, alpha=0.85)
    axes[2].set_xlabel("Mărimea populației (sol_per_pop)")
    axes[2].set_ylabel("Timp de execuție (s)")
    axes[2].set_title("Timp de execuție în funcție de mărimea populației")
    for j, (v, t) in enumerate(zip(valori_pop, durate)):
        axes[2].text(j, t + 0.01, f"{t:.2f}s", ha='center', va='bottom', fontsize=9)

    plt.suptitle("Studiu: Impactul mărimii populației", fontsize=13)
    plt.tight_layout()
    plt.show()


def studiu_parametri_mutatie():
    """
    Sarcina 3: Compară configurații cu rate de mutație diferite.
    """
    valori_mut = [5, 20, 40, 60, 80, 95]
    culori = ['navy', 'royalblue', 'steelblue', 'seagreen', 'darkorange', 'tomato']

    fig, ax = plt.subplots(figsize=(12, 6))
    distante_finale = []

    for mut, culoare in zip(valori_mut, culori):
        ga, dist, _ = ruleaza_ga(
            pop_size=100, n_generatii=300, rata_mutatie=mut, verbose=False
        )
        distante_finale.append(dist)
        d = [-f for f in ga.best_solutions_fitness]
        ax.plot(d, color=culoare, linewidth=1.5, label=f"mut={mut}%")

    ax.set_xlabel("Generație")
    ax.set_ylabel("Distanță totală")
    ax.set_title("Curbe de convergență - rate de mutație diferite")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

    print("\nRezultate studiu mutație:")
    for mut, dist in zip(valori_mut, distante_finale):
        print(f"  mut={mut:3d}%  →  distanță finală: {dist:.2f}")


def studiu_tip_selectie():
    """
    Sarcina 4: Compară strategii de selecție a părinților.
    """
    strategii = ["tournament", "rws", "rank", "sus"]
    culori = ['steelblue', 'tomato', 'seagreen', 'darkorange']

    fig, ax = plt.subplots(figsize=(12, 6))
    rezultate = []

    for strategie, culoare in zip(strategii, culori):
        ga, dist, durata = ruleaza_ga(
            pop_size=100, n_generatii=300, rata_mutatie=40,
            tip_selectie=strategie, verbose=False
        )
        rezultate.append({"strategie": strategie, "distanta": dist, "durata": durata})
        d = [-f for f in ga.best_solutions_fitness]
        ax.plot(d, color=culoare, linewidth=1.5, label=strategie)

    ax.set_xlabel("Generație")
    ax.set_ylabel("Distanță totală")
    ax.set_title("Curbe de convergență - strategii de selecție diferite")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

    print("\n{:<15} {:>15} {:>12}".format("Strategie", "Distanță finală", "Timp (s)"))
    print("-" * 45)
    for r in rezultate:
        print(f"{r['strategie']:<15} {r['distanta']:>15.2f} {r['durata']:>12.2f}")