#coding:utf8

import pandas as pd
import matplotlib.pyplot as plt

# Source des données : https://www.data.gouv.fr/datasets/election-presidentielle-des-10-et-24-avril-2022-resultats-definitifs-du-1er-tour/
with open("./data/resultats-elections-presidentielles-2022-1er-tour.csv","r") as fichier:
    contenu = pd.read_csv(fichier)

print(pd.DataFrame(contenu))

nombre_lignes = len(contenu)
nombre_colonnes = len(contenu.columns)

print("Nombre de lignes : ", nombre_lignes)
print("Nombre de colonnes : ", nombre_colonnes)

type_map = {
    'int64': 'int',
    'float64': 'float',
    'bool': 'bool',
    'object': 'str',
    'string': 'str'
}

for col in contenu.columns:
    dtype = str(contenu[col].dtype)
    type_logique = type_map.get(dtype, 'str')

print(contenu.head(0))

inscrits = contenu.loc[0, "Inscrits"]

colonnes_quantitatives = []

for col in contenu.columns:
    dtype = str(contenu[col].dtype)
    type_logique = type_map.get(dtype, "str")
    if type_logique in ["int", "float"]:
        colonnes_quantitatives.append(col)


# Affichage
for col in colonnes_quantitatives:
    print(f"{col}: {contenu[col].sum()}")

# Boucle sur les départements uniques
for dept in contenu["Code du département"].unique():
    # Filtrer les données du département
    df_dept = contenu[contenu["Code du département"] == dept]

    # Somme des inscrits et votants dans ce département
    inscrits = df_dept["Inscrits"].sum()
    votants = df_dept["Votants"].sum()

    # Créer le graphique
    plt.figure(figsize=(6, 4))
    plt.bar(["Inscrits", "Votants"], [inscrits, votants], color=["skyblue", "salmon"])
    plt.title(f"Département {dept} - Inscrits vs Votants")
    plt.ylabel("Nombre de personnes")

    # Sauvegarder dans un fichier image
    filename = f"diagrammes/departement_{dept}.png"
    plt.savefig(filename)
    plt.close()

# Boucle sur chaque département
for dept in contenu["Code du département"].unique():
    # Filtrer les données du département
    df_dept = contenu[contenu["Code du département"] == dept]
    valeurs = [
        df_dept["Abstentions"].sum(),
        df_dept["Blancs"].sum(),
        df_dept["Nuls"].sum(),
        df_dept["Exprimés"].sum()
        ]
    labels = ["Abstentions", "Blancs", "Nuls", "Exprimés"]
    valeurs_filtrees = [v for v in valeurs if v > 0]
    labels_filtres = [labels[i] for i in range(len(labels)) if valeurs[i] > 0]
    # Créer le graphique
    plt.figure(figsize=(6, 6))
    plt.pie(valeurs_filtrees, labels=labels_filtres, autopct='%1.1f%%', startangle=90)
    plt.title(f"Département {dept} - Répartition des votes")

    # Sauvegarder l’image
    filename = f"diagrammes_camembert/camembert_{dept}.png"
    plt.savefig(filename)
    plt.close()

plt.figure(figsize=(10, 6))
plt.hist(contenu["Inscrits"], bins=30, color="skyblue", edgecolor="black")
plt.title("Distribution des inscrits")
plt.xlabel("Nombre d'inscrits")
plt.ylabel("Fréquence")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Enregistrer l'image dans un fichier PNG
plt.tight_layout()
plt.savefig("distribution_inscrits.png")
plt.close()

#print()
