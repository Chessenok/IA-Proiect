import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Setăm backend-ul pentru a evita eroarea din PyCharm
import matplotlib
matplotlib.use('TkAgg')

# Încărcăm dataset-ul tips
tips = sns.load_dataset('tips')

# Creăm figura și grila de subploturi 2×2
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Analiza Dataset-ului Tips', fontsize=16, fontweight='bold')

# 1. Scatter plot (Matplotlib): total_bill vs. tip, colorat după sex
ax1 = axes[0, 0]
ax1.set_title('Total Bill vs Tip (colorat după sex)', fontsize=12)

# Iterăm prin categorii de sex pentru a crea scatter plot-ul
sex_categories = tips['sex'].unique()
colors = {'Male': 'blue', 'Female': 'red'}
markers = {'Male': 'o', 'Female': 's'}

for sex in sex_categories:
    subset = tips[tips['sex'] == sex]
    ax1.scatter(subset['total_bill'], subset['tip'],
               color=colors[sex], marker=markers[sex],
               label=sex, alpha=0.6, edgecolors='black', linewidth=0.5)

ax1.set_xlabel('Total Bill ($)')
ax1.set_ylabel('Tip ($)')
ax1.legend(title='Sex')
ax1.grid(True, alpha=0.3)

# 2. Boxplot (Seaborn): distribuția total_bill per day, în ordinea Thur → Sun
ax2 = axes[0, 1]
ax2.set_title('Distribuția Total Bill pe Zile', fontsize=12)

# Definim ordinea corectă a zilelor
day_order = ['Thur', 'Fri', 'Sat', 'Sun']

# Boxplot cu Seaborn
sns.boxplot(data=tips, x='day', y='total_bill', order=day_order, ax=ax2, palette='Set2')
ax2.set_xlabel('Ziua săptămânii')
ax2.set_ylabel('Total Bill ($)')
ax2.grid(True, alpha=0.3, axis='y')

# 3. Histogramă (Seaborn): distribuția tip, cu hue='time' și KDE suprapus
ax3 = axes[1, 0]
ax3.set_title('Distribuția Tips-urilor (Lunch vs Dinner)', fontsize=12)

# Histogramă cu KDE
sns.histplot(data=tips, x='tip', hue='time', kde=True, ax=ax3,
             palette={'Lunch': 'orange', 'Dinner': 'purple'},
             alpha=0.6, edgecolor='black', linewidth=0.5)

ax3.set_xlabel('Tip ($)')
ax3.set_ylabel('Frecvență')
ax3.legend(title='Momentul zilei', labels=['Dinner', 'Lunch'])

# 4. Barplot (Seaborn): bacșișul mediu per day, cu interval de încredere
ax4 = axes[1, 1]
ax4.set_title('Media Tips-urilor pe Zile (cu 95% CI)', fontsize=12)

# Barplot cu interval de încredere
sns.barplot(data=tips, x='day', y='tip', order=day_order, ax=ax4,
            palette='coolwarm', errorbar='ci', capsize=0.1, errwidth=1.5)

ax4.set_xlabel('Ziua săptămânii')
ax4.set_ylabel('Media Tips ($)')
ax4.grid(True, alpha=0.3, axis='y')

# Adăugăm valorile medii deasupra barelor
for i, day in enumerate(day_order):
    mean_val = tips[tips['day'] == day]['tip'].mean()
    ax4.text(i, mean_val + 0.1, f'{mean_val:.2f}$',
             ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.subplots_adjust(top=0.92)  # Facem loc pentru titlul principal

#plt.savefig('tips_analysis_grid.png', dpi=300, bbox_inches='tight')
plt.show()


print("\n" + "="*60)
print("REZUMAT STATISTIC")
print("="*60)

print("\nMedia tips-urilor pe zi:")
print(tips.groupby('day')['tip'].mean().round(2))

print("\nMedia tips-urilor pe momentul zilei:")
print(tips.groupby('time')['tip'].mean().round(2))

print("\nCorelația între total_bill și tip:")
correlation = tips['total_bill'].corr(tips['tip'])
print(f"Coeficient de corelație Pearson: {correlation:.3f}")