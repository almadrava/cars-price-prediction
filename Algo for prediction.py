#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 01:13:38 2023

@author: charlottepapelard
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# Charger les données
data = pd.read_excel("/Users/charlottepapelard/Desktop/spoti_cleaned.xlsx")


# Diviser les données en données d'entraînement et de test
X = data[['Marque', 'Année', 'Carburant', 'Kilométrage', 'Transmission']]
y = data['Prix']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)

# Faire des prédictions sur les données de test
y_pred = model.predict(X_test)

# Évaluer le modèle
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Erreur quadratique moyenne : {mse}')
print(f'Coefficient de détermination (R^2) : {r2}')



# # Collecter les caractéristiques de la nouvelle voiture à prédire
# nouvelle_voiture = {
#     'Marque': 206,
#     'Année': 2021,
#     'Carburant': 0,
#     'Kilométrage': 25845,
#     'Transmission': 0
# }

# # Transformer les caractéristiques de la nouvelle voiture en un DataFrame
# nouvelle_voiture_df = pd.DataFrame(nouvelle_voiture, index=[0])

# # Encoder les caractéristiques de la nouvelle voiture
# for col in categorical_cols:
#     nouvelle_voiture_df[col] = label_encoder.transform([nouvelle_voiture[col]])

# # Faire la prédiction du prix de la nouvelle voiture
# prix_pred = model.predict(nouvelle_voiture_df)

# # Afficher le prix prédit
# print(f'Prix prédit de la nouvelle voiture : {prix_pred[0]} €')













