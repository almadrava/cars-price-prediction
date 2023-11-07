#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 07:51:16 2023

@author: charlottepapelard
"""

############################
#### DATA PREPARATION ######
############################

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Charger les données
data = pd.read_excel("/Users/charlottepapelard/Desktop/spoticar.xlsx")

# Supprimer les doublons
data = data.drop_duplicates(subset=['Marque', 'Prix', 'Année', 'Carburant', 'Kilométrage', 'Transmission'])

# Supprimez le symbole "€" et les espaces de la colonne "Prix"
data['Prix'] = data['Prix'].str.replace(' €', '').str.replace(' ', '').astype(float)

# Supprimez les espaces et "Km" de la colonne "Kilométrage"
data['Kilométrage'] = data['Kilométrage'].str.replace(' Km', '').str.replace(' ', '')

# Convertissez la colonne "Kilométrage" en valeurs numériques
data['Kilométrage'] = data['Kilométrage'].astype(float)

# Créer un objet LabelEncoder
label_encoder = LabelEncoder()

# Appliquer l'encodage sur les colonnes catégorielles
categorical_cols = ['Marque', 'Carburant', 'Transmission']
for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])

# Extrait l'année à partir de la colonne "Année"
data['Année'] = pd.to_datetime(data['Année']).dt.year




############################
#### ALGO DE PREDICTION ####
############################


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

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




