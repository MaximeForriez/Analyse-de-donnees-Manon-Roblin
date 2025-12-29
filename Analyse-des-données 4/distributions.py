import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import (
    randint, binom, poisson, zipf, norm, lognorm,
    uniform, chi2, pareto
)

# --- Lois discrètes ---

# 1️⃣ Loi de Dirac
x = np.linspace(-5, 5, 100)
y = np.zeros_like(x)
y[np.argmin(np.abs(x - 0))] = 1
plt.figure()
plt.stem(x, y)
plt.title("Loi de Dirac (x0 = 0)")
plt.xlabel("x"); plt.ylabel("P(X=x)")
plt.grid(alpha=0.3)

# 2️⃣ Loi uniforme discrète
a, b = 1, 6
x = np.arange(a, b+1)
pmf = randint.pmf(x, a, b+1)
plt.figure()
plt.stem(x, pmf)
plt.title(f"Loi uniforme discrète U({a},{b})")
plt.xlabel("x"); plt.ylabel("P(X=x)")
plt.grid(alpha=0.3)

# 3️⃣ Loi binomiale
n, p = 10, 0.5
x = np.arange(0, n+1)
pmf = binom.pmf(x, n, p)
plt.figure()
plt.stem(x, pmf)
plt.title(f"Loi binomiale B({n},{p})")
plt.xlabel("x"); plt.ylabel("P(X=x)")
plt.grid(alpha=0.3)

# 4️⃣ Loi de Poisson (discrète)
lam = 3
x = np.arange(0, 15)
pmf = poisson.pmf(x, lam)
plt.figure()
plt.stem(x, pmf)
plt.title(f"Loi de Poisson (λ={lam})")
plt.xlabel("x"); plt.ylabel("P(X=x)")
plt.grid(alpha=0.3)

# 5️⃣ Loi de Zipf-Mandelbrot
a = 2
x = np.arange(1, 10)
pmf = zipf.pmf(x, a)
plt.figure()
plt.stem(x, pmf)
plt.title(f"Loi de Zipf (a={a})")
plt.xlabel("x"); plt.ylabel("P(X=x)")
plt.grid(alpha=0.3)


# --- Lois continues ---

# 6️⃣ Loi normale
mu, sigma = 0, 1
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 300)
pdf = norm.pdf(x, mu, sigma)
plt.figure()
plt.plot(x, pdf)
plt.title(f"Loi normale N({mu},{sigma**2})")
plt.xlabel("x"); plt.ylabel("f(x)")
plt.grid(alpha=0.3)

# 7️⃣ Loi log-normale
s, scale = 0.954, np.exp(0)
x = np.linspace(0, 5, 300)
pdf = lognorm.pdf(x, s, scale=scale)
plt.figure()
plt.plot(x, pdf)
plt.title(f"Loi log-normale (s={s})")
plt.xlabel("x"); plt.ylabel("f(x)")
plt.grid(alpha=0.3)

# 8️⃣ Loi uniforme continue
a, b = 0, 1
x = np.linspace(a - 0.2, b + 0.2, 300)
pdf = uniform.pdf(x, a, b - a)
plt.figure()
plt.plot(x, pdf)
plt.title(f"Loi uniforme continue U({a},{b})")
plt.xlabel("x"); plt.ylabel("f(x)")
plt.grid(alpha=0.3)

# 9️⃣ Loi du χ²
df = 5
x = np.linspace(0, 20, 300)
pdf = chi2.pdf(x, df)
plt.figure()
plt.plot(x, pdf)
plt.title(f"Loi du χ² (ddl={df})")
plt.xlabel("x"); plt.ylabel("f(x)")
plt.grid(alpha=0.3)

# 🔟 Loi de Pareto
b = 3
x = np.linspace(1, 10, 300)
pdf = pareto.pdf(x, b)
plt.figure()
plt.plot(x, pdf)
plt.title(f"Loi de Pareto (b={b})")
plt.xlabel("x"); plt.ylabel("f(x)")
plt.grid(alpha=0.3)

# Affiche toutes les figures
plt.show()

# Fonction pour calculer la moyenne et l'écart type

def calcul_moyenne_et_ecart():
    resultats = {}

    # --- Loi de Dirac ---
    x0 = 0
    resultats["Dirac"] = {"moyenne": x0, "ecart_type": 0}

    # --- Lois discrètes ---
    resultats["Uniforme discrète"] = {"moyenne": randint.mean(1, 7),
                                      "ecart_type": randint.std(1, 7)}

    resultats["Binomiale"] = {"moyenne": binom.mean(n=10, p=0.5),
                              "ecart_type": binom.std(n=10, p=0.5)}

    resultats["Poisson"] = {"moyenne": poisson.mean(mu=3),
                             "ecart_type": poisson.std(mu=3)}

    resultats["Zipf-Mandelbrot"] = {"moyenne": zipf.mean(a=2),
                                     "ecart_type": zipf.std(a=2)}

    # --- Lois continues ---
    resultats["Normale"] = {"moyenne": norm.mean(loc=0, scale=1),
                             "ecart_type": norm.std(loc=0, scale=1)}

    resultats["Log-normale"] = {"moyenne": lognorm.mean(s=0.954, scale=np.exp(0)),
                                 "ecart_type": lognorm.std(s=0.954, scale=np.exp(0))}

    resultats["Uniforme continue"] = {"moyenne": uniform.mean(loc=0, scale=1),
                                      "ecart_type": uniform.std(loc=0, scale=1)}

    resultats["Chi2"] = {"moyenne": chi2.mean(df=5),
                         "ecart_type": chi2.std(df=5)}

    resultats["Pareto"] = {"moyenne": pareto.mean(b=3),
                            "ecart_type": pareto.std(b=3)}
    
    print(resultats)

calcul_moyenne_et_ecart()

