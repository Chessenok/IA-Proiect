import matplotlib.pyplot as plt
from problem import COORD, N_ORASE, ORASE, distanta_ruta

# ══════════════════════════════════════════════════════════════════
# 6. VIZUALIZARE
# ══════════════════════════════════════════════════════════════════

def plot_convergenta(ga_instance, titlu="Curba de convergență", ax=None):
    """
    Grafic: distanța celei mai bune soluții per generație (curba de convergență).
    Dacă ax este None, creează o figură nouă și o afișează.
    """
    distante = [-f for f in ga_instance.best_solutions_fitness]
    afiseaza_singur = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(distante, color='steelblue', linewidth=1.5, label='Cea mai bună soluție')
    ax.set_xlabel("Generație")
    ax.set_ylabel("Distanță totală")
    ax.set_title(titlu)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if afiseaza_singur:
        plt.tight_layout()
        plt.show()


def plot_ruta(solutie, titlu="Ruta găsită de AG"):
    """Grafic: harta orașelor cu ruta vizualizată prin săgeți."""
    solutie_int = [int(c) for c in solutie]
    ruta = solutie_int + [solutie_int[0]]

    fig, ax = plt.subplots(figsize=(12, 9))

    for i in range(len(ruta) - 1):
        x1, y1 = COORD[ruta[i]]
        x2, y2 = COORD[ruta[i + 1]]
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="steelblue", lw=2)
        )
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my, str(i + 1), fontsize=7, color='gray', ha='center')

    ax.scatter(COORD[:, 0], COORD[:, 1], s=150, c='tomato', zorder=5)
    for i in range(N_ORASE):
        x, y = COORD[i]
        ax.annotate(ORASE[i][0], (x, y),
                    textcoords="offset points", xytext=(10, 5), fontsize=9)

    distanta = distanta_ruta(solutie)
    ax.set_title(f"{titlu}\nDistanță totală: {distanta:.2f}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()