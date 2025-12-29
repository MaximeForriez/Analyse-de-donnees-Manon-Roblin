# coding:utf8

import pandas as pd
import numpy as np
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import os

dossier_data = "/Users/manonroblin/Documents/projet/data"
chemin_image = os.path.join(dossier_data, "mon_image.png")

def ouvrirUnFichier(path):
    return pd.read_csv(path)

def intervalle_fluctuation(p, n, z=1.96):
    margin = z * np.sqrt(p * (1 - p) / n)
    return (round(p - margin, 3), round(p + margin, 3))

plt.plot([1, 2, 3], [4, 5, 6], marker='o', linestyle='-', color='green')
plt.title("Mon graphique")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.savefig(chemin_image)
plt.show()
plt.close()
print(f"Image sauvegardée dans {chemin_image}\n")

print("=== Théorie de l'échantillonnage ===\n")

df = ouvrirUnFichier(os.path.join(dossier_data, "Echantillonnage-100-Echantillons.csv"))
print("Aperçu du fichier :\n", df.head(), "\n")

moyennes = df.mean().round(0).astype(int)
print("Moyennes arrondies :\n", moyennes, "\n")

total = moyennes.sum()
frequences = (moyennes / total).round(2)
print("Fréquences des moyennes :\n", frequences, "\n")
pop = pd.Series({"Pour": 852, "Contre": 911, "SansOpinion": 422})
frequences_pop = (pop / pop.sum()).round(2)
print("Fréquences population mère :\n", frequences_pop, "\n")

intervales = {opinion: intervalle_fluctuation(freq, total)
              for opinion, freq in frequences.items()}
print("Intervalles de fluctuation :\n", intervales, "\n")

print("=== Théorie de l'estimation ===\n")

ligne = df.iloc[0]
n_ech = ligne.sum()
freqs_ech = (ligne / n_ech).round(2)
print("Fréquences du premier échantillon :\n", freqs_ech, "\n")

IC = {op: intervalle_fluctuation(freq, n_ech) for op, freq in freqs_ech.items()}
print("Intervalles de confiance du premier échantillon :\n", IC, "\n")

print("=== Théorie de la décision ===\n")

test1 = ouvrirUnFichier(os.path.join(dossier_data, "Loi-normale-Test-1.csv"))
test2 = ouvrirUnFichier(os.path.join(dossier_data, "Loi-normale-Test-2.csv"))

stat1, p1 = shapiro(test1.iloc[:, 0])
stat2, p2 = shapiro(test2.iloc[:, 0])

print("Test de Shapiro-Wilk :")
print(f"Test 1 : stat={stat1:.4f}, p-value={p1:.4f}")
print(f"Test 2 : stat={stat2:.4f}, p-value={p2:.4f}\n")

chemin_test2 = os.path.join(dossier_data, "hist_test2.png")
plt.hist(test2.iloc[:, 0], bins=15, color='skyblue', edgecolor='black')
plt.title("Distribution du fichier Test 2")
plt.xlabel("Valeurs")
plt.ylabel("Effectif")
plt.grid(axis='y', alpha=0.75)
plt.savefig(chemin_test2)
plt.show()
plt.close()