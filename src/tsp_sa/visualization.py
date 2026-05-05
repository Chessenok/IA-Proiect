import matplotlib.pyplot as plt
import os

def plot_tour(cities: list, tour: list[int], title: str = "TSP Tour"):
    coords = [cities[i] for i in tour] + [cities[tour[0]]]
    xs, ys = zip(*coords)
    
    plt.figure(figsize=(8, 6))
    plt.plot(xs, ys, 'b-o', markersize=8, linewidth=1.5)
    
    for idx, (x, y) in enumerate(cities):
        plt.annotate(str(idx), (x, y), textcoords="offset points",
                     xytext=(5, 5), fontsize=9)
    
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = f"{title.replace(' ', '_')}.png"
    save_path = os.path.join("Results Charts", "Lab8", filename)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=150)
    plt.show()