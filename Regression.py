#### Chargement des packages ####

# Importation des bibliothèques nécessaires
import pandas as pd  # Pour la manipulation des données
import statsmodels.api as sm  # Pour la modélisation statistique
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # Pour la préparation des données
from sklearn.linear_model import LinearRegression  # Pour l'ajustement du modèle
from sklearn.model_selection import train_test_split  # Pour la division des données

#### Chargement des données ####

# Chargement des données depuis un fichier Excel
data = pd.read_excel("/Users/charlottepapelard/Desktop/spoticar.xlsx")

#### Définition des variables quantitatives et qualitatives ####

# Définition des noms de colonnes considérées comme variables quantitatives
quantitative_features = ['Kilometrage', 'Annee']

# Définition des noms de colonnes considérées comme variables qualitatives
qualitative_features = ['Carburant', 'Transmission', 'Marque']

#### Prétraitement des données ####

# Renommer les colonnes pour une meilleure compréhension
data = data.rename(columns={'Marque': 'Modele'})
data = data.rename(columns={'Kilométrage': 'Kilometrage'})
data = data.rename(columns={'Année': 'Annee'})

# Extraire la marque à partir de la colonne "Modele"
data['Marque'] = data['Modele'].apply(lambda x: x.split(' ', 1)[0])

# Nettoyer et convertir les colonnes pertinentes en formats appropriés
data['Kilometrage'] = data['Kilometrage'].str.replace(r'\D', '', regex=True).astype(int)
data['Prix'] = data['Prix'].str.replace(r'\D', '', regex=True).astype(int)
data['Annee'] = data['Annee'].dt.year

#### Division des données en un ensemble d'entraînement ####

# Création d'un DataFrame de test en excluant la colonne "Prix"
test = data.drop("Prix", axis=1)

# Séparation des prédicteurs (X) et de la variable cible (y)
X = test.drop("Modele", axis=1)
y = data["Prix"]

# Ajout d'une constante à X pour la régression
X = sm.add_constant(X)

# Standardisation des variables quantitatives
scaler = StandardScaler()
X[quantitative_features] = scaler.fit_transform(X[quantitative_features])

# Encodage one-hot des variables qualitatives
encoder = OneHotEncoder(sparse=False, drop='first')
Xquali = encoder.fit_transform(X[qualitative_features])
Xquali_df = pd.DataFrame(Xquali, columns=encoder.get_feature_names(qualitative_features))

# Concaténation des données quantitatives et des données encodées
X_encoded = pd.concat([X[quantitative_features], Xquali_df], axis=1)

#### Division des données en ensembles d'entraînement et de test ####

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Ajustement du modèle de régression linéaire sur l'ensemble d'entraînement
model = LinearRegression()
model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Obtention des coefficients et de l'interception du modèle
coefficient = model.coef_
intercept = model.intercept_

# Évaluation du modèle en calculant le score R2
score = model.score(X_test, y_test)

# Impression des résultats
print("Coefficient:", coefficient)
print("Intercept:", intercept)
print("Score R2:", score)

