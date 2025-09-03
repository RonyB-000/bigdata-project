
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
