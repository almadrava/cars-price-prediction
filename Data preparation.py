#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 07:51:16 2023

@author: charlottepapelard
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Charger les données
data = pd.read_excel("/Users/charlottepapelard/Desktop/spoti.xlsx")

# Supprimer les doublons
data = data.drop_duplicates(subset=['Marque', 'Prix', 'Style', 'Année', 'Carburant', 'Kilométrage', 'Transmission'])

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




# Spécifiez le nom du fichier Excel de sortie
output_file = "spoti_cleaned.xlsx"

# Sauvegardez le DataFrame dans un nouveau fichier Excel
data.to_excel(output_file, index=False)

# Affichez un message de confirmation
print(f"Les données ont été sauvegardées dans {output_file}.")







