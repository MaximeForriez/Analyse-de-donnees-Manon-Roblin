#coding:utf8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Source des données : https://www.data.gouv.fr/datasets/election-presidentielle-des-10-et-24-avril-2022-resultats-definitifs-du-1er-tour/

# Sources des données : production de M. Forriez, 2016-2023

with open("data/resultats-elections-presidentielles-2022-1er-tour.csv", encoding='utf-8') as fichier:
    df = pd.read_csv(fichier) 


type_map = {
    'int64': 'int',
    'float64': 'float',
    'bool': 'bool',
    'object': 'str',
    'string': 'str'
}

colonnes_quantitatives = []

for col in df.columns:
    dtype = str(df[col].dtype)
    type_logique = type_map.get(dtype, "str")
    if type_logique in ["int", "float"]:
        colonnes_quantitatives.append(col)

liste_moyenne = []

# Affichage
for col in colonnes_quantitatives:
    moyenne = round(df[col].mean(), 2)
    liste_moyenne.append(f"{col}: {moyenne}")
print(liste_moyenne)

liste_median = []

for col in colonnes_quantitatives:
    median = df[col].median()
    liste_median.append(f"{col}: {median}")
print(liste_median)

liste_mode = []

for col in colonnes_quantitatives:
    mode_val= df[col].mode().tolist()
    liste_mode.append(f"{col}: {mode_val}")
print(liste_mode)

liste_ecart_type = []

for col in colonnes_quantitatives:
    ecart_type = round(df[col].std(), 2)
    liste_ecart_type.append(f"{col}: {ecart_type}")
print(liste_ecart_type)

liste_ecart_absolu = []

for col in colonnes_quantitatives:
    moyenne = df[col].mean()
    ecart_absolu = round(np.abs(df[col] - moyenne).mean(), 2)
    liste_ecart_absolu.append(f"{col}: {ecart_absolu}")
print(liste_ecart_absolu)

liste_etendue = []

for col in colonnes_quantitatives:
    etendue = round(df[col].max() - df[col].min(), 2)
    liste_etendue.append(f"{col}: {etendue}")
print(liste_etendue)

liste_interquartile = []

for col in colonnes_quantitatives:
    q3 = df[col].quantile(0.75)
    q1 = df[col].quantile(0.25)
    iqr = round(q3 - q1, 2)
    liste_interquartile.append(f"{col}: {iqr}")

liste_interdecile = []

for col in colonnes_quantitatives:
    d9 = df[col].quantile(0.9)
    d1 = df[col].quantile(0.1)
    idr = round(d9 - d1, 2)
    liste_interdecile.append(f"{col}: {idr}")

#for col in colonnes_quantitatives:
#    plt.figure(figsize=(6, 4))  
#    plt.boxplot(df[col], vert=True, patch_artist=True)  
#    plt.title(f'Boîte à moustaches - {col}')
#    plt.ylabel(col)
#    plt.savefig(f'img/boxplot_{col}.png', dpi=150, bbox_inches='tight')
#    plt.close()

with open("data/island-index.csv", encoding="utf-8") as fichier:
    df = pd.read_csv(fichier)

# Sélection et catégorisation de la colonne "Surface (km2)"

liste_surface = []

for surface in df["Surface (km²)"]:
    if 0 < surface <= 10:
        categorie = "]0, 10]"
    elif 10 < surface <= 25:
        categorie = "]10, 25]"
    elif 25 < surface <= 50:
        categorie = "]25, 50]"
    elif 50 < surface <= 100:
        categorie = "]50, 100]"
    elif 100 < surface <= 2500:
        categorie = "]100, 2500]"
    elif 2500 < surface <= 5000:
        categorie = "]2500, 5000]"
    elif 5000 < surface <= 10000:
        categorie = "]5000, 10000]"
    else:  # surface > 10000
        categorie = "]10000, +∞["

    liste_surface.append(categorie)

# Ajouter la liste comme nouvelle colonne dans le DataFrame
df["Classe_Surface"] = liste_surface

# Dénombrement des îles par catégorie
liste_comptage = df["Classe_Surface"].value_counts().sort_index()

plt.figure(figsize=(8, 5))
plt.bar(liste_comptage.index, liste_comptage.values, color='skyblue', edgecolor='black')

plt.title("Répartition des îles par intervalle de surface")
plt.xlabel("Intervalle de surface (km²)")
plt.ylabel("Nombre d'îles")
plt.xticks(rotation=45, ha='right')

# Enregistrement de l'organigramme dans le dossier img
plt.tight_layout()
plt.savefig("img/organigramme_surface.png", dpi=150)
plt.close()

print("=== Nombre d'îles par intervalle de surface ===") 
for categorie, nombre in liste_comptage.items(): 
    print(f"{categorie}: {nombre}")
