import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder

#################### Arangement de la base de donées ###############

# Spécifiez le chemin vers votre fichier Excel
excel_file = "spoticar.xlsx"

# Chargez le fichier Excel dans un DataFrame
car_data = pd.read_excel(excel_file)

# Renommez la colonne "Marque" en "Modele" dans le DataFrame car_data
car_data = car_data.rename(columns={'Marque': 'Modele'})

# Renommez la colonne "Kilométrage" en "Kilometrage" dans le DataFrame car_data
car_data = car_data.rename(columns={'Kilométrage': 'Kilometrage'})

# Renommez la colonne "Année" en "Annee" dans le DataFrame car_data
car_data = car_data.rename(columns={'Année': 'Annee'})

# Créer une nouvelle colonne "Marque" en extrayant la marque à partir de la colonne "Modele"
car_data['Marque'] = car_data['Modele'].apply(lambda x: x.split(' ', 1)[0])

# Supprimer les caractères indésirables et convertir 'Kilometrage' en format numérique
car_data['Kilometrage'] = car_data['Kilometrage'].str.replace(r'\D', '', regex=True).astype(int)

# Supprimer les caractères indésirables et convertir 'Prix' en format numérique
car_data['Prix'] = car_data['Prix'].str.replace(r'\D', '', regex=True).astype(int)

# Ajustez l'option d'affichage pour afficher toutes les informations
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

print(car_data.describe)
print(car_data.dtypes)

################# Statistiques descriptives Uni-variées & Bi-variées########

# Statistiques descriptives univariées pour les colonnes numériques
univariate_stats_numeric = car_data.describe(include='number')


# Sélectionner uniquement les colonnes numériques
numeric_columns = car_data.select_dtypes(include=['int', 'float'])

# Calculer la matrice de corrélation entre les colonnes numériques
bivariate_stats_numeric = numeric_columns.corr()


# Statistiques descriptives pour les colonnes non numériques
univariate_stats_non_numeric = car_data.describe(exclude=[np.number])

# Afficher les statistiques descriptives univariées pour les colonnes numériques
print("Statistiques descriptives univariées pour les colonnes numériques :")
print(univariate_stats_numeric)

# Afficher les statistiques descriptives bivariées pour les colonnes numériques
print("Statistiques descriptives bivariées (corrélation) pour les colonnes numériques :")
print(bivariate_stats_numeric)

# Afficher les statistiques descriptives pour les colonnes non numériques
print("Statistiques descriptives pour les colonnes non numériques :")
print(univariate_stats_non_numeric)


############################

# Individus uniques pour la colonne 'Marque'
marques_uniques = car_data['Marque'].unique()

# Individus uniques pour la colonne 'Transmission'
transmission_uniques = car_data['Transmission'].unique()

# Individus uniques pour la colonne 'Carburant'
carburant_uniques = car_data['Carburant'].unique()

print("Individus uniques pour la colonne 'Marque':", marques_uniques)
print("Individus uniques pour la colonne 'Transmission':", transmission_uniques)
print("Individus uniques pour la colonne 'Carburant':", carburant_uniques)

###################################
# Création de graphiques de base pour visualiser les données

# Créer un DataFrame regroupant les moyennes de prix par marque
mean_prices_by_marque = car_data.groupby('Marque')['Prix'].mean().reset_index()

# Afficher le graphique de la corrélation entre les marques et le prix
plt.figure(figsize=(12, 6))
plt.bar(mean_prices_by_marque['Marque'], mean_prices_by_marque['Prix'])
plt.xlabel('Marque')
plt.ylabel('Prix moyen')
plt.title('Corrélation entre les marques et le prix moyen')
plt.xticks(rotation=90)
plt.show()

# Créer un DataFrame regroupant les moyennes de prix par type de transmission
mean_prices_by_transmission = car_data.groupby('Transmission')['Prix'].mean().reset_index()

# Afficher le graphique de la corrélation entre la transmission et le prix
plt.figure(figsize=(8, 6))
plt.bar(mean_prices_by_transmission['Transmission'], mean_prices_by_transmission['Prix'])
plt.xlabel('Transmission')
plt.ylabel('Prix moyen')
plt.title('Corrélation entre la transmission et le prix moyen')
plt.show()

# Créer un DataFrame regroupant les moyennes de prix par type de carburant
mean_prices_by_carburant = car_data.groupby('Carburant')['Prix'].mean().reset_index()

# Afficher le graphique de la corrélation entre le carburant et le prix
plt.figure(figsize=(10, 6))
plt.bar(mean_prices_by_carburant['Carburant'], mean_prices_by_carburant['Prix'])
plt.xlabel('Carburant')
plt.ylabel('Prix moyen')
plt.title('Corrélation entre le carburant et le prix moyen')
plt.xticks(rotation=45)
plt.show()

# Extraire l'année de la colonne 'Annee'
car_data['Annee'] = car_data['Annee'].dt.year

# Créer un DataFrame regroupant les prix moyens par année
mean_prices_by_year = car_data.groupby('Annee')['Prix'].mean().reset_index()

# Afficher l'histogramme de la corrélation entre l'année et le prix moyen
plt.figure(figsize=(10, 6))
plt.bar(mean_prices_by_year['Annee'], mean_prices_by_year['Prix'])
plt.xlabel('Année')
plt.ylabel('Prix moyen')
plt.title('Corrélation entre l\'année et le prix moyen')
plt.xticks(rotation=45)
plt.show()

####################################################

# Sélectionnez les colonnes Kilometrage et Prix
X = car_data[['Kilometrage']]
Y = car_data['Prix']

# Créez un modèle de régression linéaire
model = LinearRegression()

# Ajustez le modèle aux données
model.fit(X, Y)

# Faites des prédictions sur les données originales
predictions = model.predict(X)

# Affichez le graphique des données et de la ligne de régression
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, label='Données réelles')
plt.plot(X, predictions, color='red', linewidth=2, label='Régression linéaire')
plt.xlabel('Kilometrage')
plt.ylabel('Prix')
plt.title('Régression linéaire : Prix en fonction du Kilométrage')
plt.legend()
plt.show()

# Sauvegarder la base de données mise à jour
#car_data.to_excel("Car_data.xlsx", index=False)

# Créez des tranches pour le kilométrage
bins = [0, 10000, 20000, 30000, 40000, 50000, 100000, np.inf]
labels = ['0-10k', '10-20k', '20-30k', '30-40k', '40-50k', '50-100k', '100k+']

# Ajoutez une nouvelle colonne 'Kilometrage_tranche' aux données
car_data['Kilometrage_tranche'] = pd.cut(car_data['Kilometrage'], bins=bins, labels=labels)

# Créez un histogramme pour le kilométrage par tranche
plt.figure(figsize=(10, 6))
car_data['Kilometrage_tranche'].value_counts().plot(kind='bar', color='skyblue')
plt.xlabel('Tranche de Kilométrage')
plt.ylabel('Nombre de Voitures')
plt.title('Répartition des Voitures par Tranche de Kilométrage')
plt.xticks(rotation=45)
plt.show()

#####################prediction########################


# Sélection des caractéristiques et de la variable cible
features = ['Kilometrage', 'Annee', 'Carburant', 'Transmission', 'Marque', 'Modele']
target = 'Prix'

# Créez un sous-ensemble de données avec seulement les caractéristiques catégorielles
categorical_features = ['Carburant', 'Transmission', 'Modele', 'Marque']
X_categorical = car_data[categorical_features]

# Créez une instance de OneHotEncoder
encoder = OneHotEncoder(drop='first', sparse_output=False)


# Ajustez l'encodeur aux données d'entraînement
encoder.fit(X_categorical)

# Transformez les caractéristiques catégorielles en variables indicatrices
X_categorical_encoded = encoder.transform(X_categorical)

# Créez un DataFrame avec les caractéristiques encodées
X_encoded = pd.DataFrame(X_categorical_encoded, columns=encoder.get_feature_names_out(categorical_features))

# Supprimez les colonnes originales des caractéristiques catégorielles
car_data = car_data.drop(categorical_features, axis=1)

# Concaténez les caractéristiques encodées avec le reste des caractéristiques numériques
car_data = pd.concat([car_data, X_encoded], axis=1)

# Sélection des caractéristiques et de la variable cible
features = ['Kilometrage', 'Annee'] + list(encoder.get_feature_names_out(categorical_features))
target = 'Prix'

# Diviser les données en ensemble d'entraînement et ensemble de test
X = car_data[features]
y = car_data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)

# Prédire les prix
y_pred = model.predict(X_test)

# Évaluation du modèle
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R² : {r2:.2f}")
print(f"RMSE : {rmse:.2f}")