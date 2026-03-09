import seaborn as sns
import pandas as pd


tips = sns.load_dataset("tips")
#print(tips.head(10))
print(tips.info())
mean_tips = tips.groupby(['day', 'sex'])['tip'].mean(numeric_only=True)
print(mean_tips)

tips["procentaj"] = (tips["tip"] / tips["total_bill"]) * 100

print(tips.head(10))

maxs = tips["procentaj"].sort_values(ascending=False)
print(f'cele mai generoase mese(dupa %) - {maxs.head(5)}')

smokers = tips.groupby(['day', 'smoker'])['size'].mean().sort_values(ascending=False)
print(f'Mese fumatori/nu - {smokers}')

