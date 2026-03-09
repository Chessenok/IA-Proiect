import seaborn as sns
import matplotlib.pyplot as plt

# Incarcare date
iris = sns.load_dataset('iris')

# Pairplot dataset Iris
g = sns.pairplot(iris, hue='species', diag_kind='kde', palette='viridis')
plt.suptitle('Analiza Pairplot a Setului de Date Iris', y=1.05, fontsize=16)
plt.savefig('iris_pairplot.png', bbox_inches='tight')
print("Figura 'iris_pairplot.png' a fost salvată.")


# Violinplots per variabila numerica
fig, axes = plt.subplots(1, 4, figsize=(20, 6))
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

for i, feature in enumerate(features):
    sns.violinplot(
        data=iris,
        x='species', 
        y=feature, 
        hue='species', 
        split=False, 
        ax=axes[i], 
        palette='muted',
        legend=False
    )
    axes[i].set_title(f'Distribuția {feature.replace("_", " ").title()}')
    axes[i].set_xlabel('Specie')
    axes[i].set_ylabel('Valoare (cm)')

plt.suptitle('Distribuția Variabilelor Iris prin Violinplots', fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig('iris_violinplots.png')
print("Figura 'iris_violinplots.png' a fost salvată.")

plt.show()
