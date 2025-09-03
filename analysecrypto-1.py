
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import yfinance as yf

#  Liste des cryptos à analyser (tickers Yahoo Finance)
cryptos = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD', 'XRP-USD', 'AVAX-USD']

#  Définition de la période longue (7 ans ici)
start_date = "2018-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')

#  Téléchargement des prix ajustés journaliers via yfinance

data = yf.download(cryptos, start=start_date, end=end_date)
data = data['Close']  # Remplace 'Adj Close' par 'Close'


#  Nettoyage des données : suppression des lignes contenant des NaN
data = data.dropna()

#  Calcul des rendements log quotidiens
returns = np.log(data / data.shift(1)).dropna()

#  Sauvegarde locale pour éviter de recharger à chaque fois
data.to_csv("crypto_prices.csv")
returns.to_csv("crypto_returns.csv")

#  Affichage d'un aperçu
print("Prix ajustés (dernières lignes) :")
print(data.tail())

print("\nRendements log (dernières lignes) :")
print(returns.tail())

#etape 2

# Rendements cumulés (log cumulés exponentiés)
cumulative_returns = (returns.cumsum()).apply(np.exp)

plt.figure(figsize=(14, 7))
for col in cumulative_returns.columns:
    plt.plot(cumulative_returns.index, cumulative_returns[col], label=col)
plt.title('Rendements cumulés des cryptos depuis 2018')
plt.ylabel('Croissance (base 1)')
plt.xlabel('Date')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# etape 3
# Corrélation des rendements log
correlation_matrix = returns.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Corrélation des rendements log (quotidiens)')
plt.tight_layout()
plt.show()

# etape 4
plt.figure(figsize=(10, 6))
sns.boxplot(data=returns, orient="h")
plt.title("Distribution des rendements quotidiens")
plt.xlabel("Rendement log")
plt.grid(True)
plt.tight_layout()
plt.show()

#etape 5
sharpe_ratios = (returns.mean() / returns.std()) * np.sqrt(365)
sharpe_ratios = sharpe_ratios.sort_values(ascending=False)

print("📈 Ratios de Sharpe annualisés (risk-free rate = 0) :")
print(sharpe_ratios.round(2))

# Barplot
plt.figure(figsize=(10, 5))
sharpe_ratios.plot(kind='bar', color='mediumseagreen')
plt.title("Ratios de Sharpe annualisés (sans risque)")
plt.ylabel("Sharpe Ratio")
plt.grid(True)
plt.tight_layout()
plt.show()


#Étape 1 : Préparation des données pour l'optimisation
import scipy.optimize as sco

# Rendements moyens annuels et matrice de covariance annualisée
mean_returns = returns.mean() * 365
cov_matrix = returns.cov() * 365
num_assets = len(mean_returns)
assets = mean_returns.index

#Étape 2 : Fonctions pour optimiser
# Fonction de performance : rendement et volatilité du portefeuille
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, std

# Fonction à minimiser : -Sharpe Ratio (car on veut le maximiser)
def neg_sharpe_ratio(weights, mean_returns, cov_matrix):
    p_return, p_vol = portfolio_performance(weights, mean_returns, cov_matrix)
    return -p_return / p_vol

# Fonction à minimiser pour la frontière efficiente : la volatilité
def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_performance(weights, mean_returns, cov_matrix)[1]

#Étape 3 : Contraintes & limites
# Contraintes : somme des poids = 1
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
# Limites : chaque poids entre 0 et 1 (pas de ventes à découvert ici)
bounds = tuple((0, 1) for _ in range(num_assets))

# Poids initiaux aléatoires
init_guess = num_assets * [1. / num_assets]

#Étape 4 : Optimisation pour max Sharpe Ratio
opt_sharpe = sco.minimize(neg_sharpe_ratio, init_guess,
                          args=(mean_returns, cov_matrix),
                          method='SLSQP', bounds=bounds,
                          constraints=constraints)

optimal_weights = opt_sharpe.x
opt_return, opt_vol = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
opt_sharpe_ratio = opt_return / opt_vol

print("📊 Portefeuille Optimal (Sharpe Ratio max) :")
for asset, weight in zip(assets, optimal_weights):
    print(f"{asset}: {weight:.2%}")
print(f"\nRendement espéré : {opt_return:.2%}")
print(f"Volatilité : {opt_vol:.2%}")
print(f"Sharpe Ratio : {opt_sharpe_ratio:.2f}")

#Étape 5 : Simulation de la frontière efficiente
n_portfolios = 5000
results = np.zeros((3, n_portfolios))
weights_record = []

for i in range(n_portfolios):
    weights = np.random.dirichlet(np.ones(num_assets), size=1)[0]
    weights_record.append(weights)
    portfolio_return, portfolio_vol = portfolio_performance(weights, mean_returns, cov_matrix)
    results[0, i] = portfolio_vol
    results[1, i] = portfolio_return
    results[2, i] = portfolio_return / portfolio_vol  # Sharpe Ratio

# Extraction des meilleurs portfolios
max_sharpe_idx = np.argmax(results[2])
sdp, rp = results[0, max_sharpe_idx], results[1, max_sharpe_idx]
max_sr_weights = weights_record[max_sharpe_idx]

#Étape 6 : Visualisation de la frontière efficiente
plt.figure(figsize=(12, 8))
plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap='viridis', marker='o', s=10, alpha=0.3)
plt.colorbar(label='Sharpe Ratio')
plt.scatter(sdp, rp, marker='*', color='red', s=300, label='Portefeuille optimal (Sharpe max)')
plt.title('Frontière Efficiente - Portefeuille Crypto')
plt.xlabel('Volatilité')
plt.ylabel('Rendement espéré')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
