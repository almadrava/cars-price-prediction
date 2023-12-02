
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import pickle
import os
import sys


data = pd.read_excel(r"C:\Users\Administrateur\Desktop\clean_data.xlsx")
# Séparez les variables indépendantes (X) et la variable dépendante (y)
X = data[["marque", "cylindree" , "categorie" , "annee" ,"boite_vitesse" ,"carburant" ,"kilometrage" ,"nb_places" , "nb_portes" , "nb_vitesses" ,"puissance_fiscale" ,"puissance_physique"]]
y = data['prix']



# Assuming X is your original DataFrame
X_encoded = X.copy()

# Dictionary to store the encoders
categorical_encoders = {}
quantitative_encoders = {}

# Iterating over each column of X
for column in X.columns:
    if X[column].dtype == "object":
        frequency = X[column].value_counts(normalize=True)
        X_encoded[column] = X[column].map(frequency)
        categorical_encoders[column] = frequency
    else:
        # Simple Standard scaling for quantitative variables
        ss = StandardScaler()
        X_encoded[column] = ss.fit_transform(X[[column]])
        quantitative_encoders[column] = {'scaler': ss, 'mean': X[column].mean(), 'std': X[column].std()}

# Saving the encoders separately with pickle
with open('categorical_encoders.pkl', 'wb') as file:
    pickle.dump(categorical_encoders, file)

with open('quantitative_encoders.pkl', 'wb') as file:
    pickle.dump(quantitative_encoders, file)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Entraîner votre modèle de prédiction (Random Forest) sur X_train_grouped, y_cost_train_grouped et y_revenue_train_grouped
rf_model = RandomForestRegressor(max_depth= 15, min_samples_split= 2, n_estimators= 200)
rf_model.fit(X_train, y_train)


# Utiliser le modèle entraîné (Random Forest) pour effectuer des prédictions sur les données de test (X_test)
#y_pred = rf_model.predict(X_test)

# Évaluer les performances de votre modèle (Random Forest) en utilisant les mesures d'évaluation appropriées (par exemple, MSE, MAE, R2)
#mse_rf = mean_squared_error(y_test, y_pred)
#mae_rf = mean_absolute_error(y_test, y_pred)
#r2_rf= r2_score(y_test, y_pred)
#rmse_rf = mean_squared_error(y_test, y_pred, squared=False)
#print("RMSE:", rmse_rf)

#print("Mean Squared Error (MSE):", mse_rf)
#print("Mean Absolute Error (MAE):", mae_rf)
#print("R-squared (R2):", r2_rf)

with open('rf_model.pkl', 'wb') as file:
    pickle.dump(rf_model, file)



def estimate_car_price(marque, cylindree, categorie, annee, boite_vitesse, carburant,
                       kilometrage, nb_places, nb_portes, nb_vitesses, puissance_fiscale, puissance_physique):
    # Load categorical encoders and the model
    with open('categorical_encoders.pkl', 'rb') as file:
        categorical_encoders = pickle.load(file)
    with open('rf_model.pkl', 'rb') as file:
        rf_model = pickle.load(file)

    # Map categorical values using encoders
    marque = categorical_encoders['marque'].get(marque, 0)  # Replace 0 with default value if needed
    categorie = categorical_encoders['categorie'].get(categorie, 0)  # Replace 0 with default value if needed
    boite_vitesse = categorical_encoders['boite_vitesse'].get(boite_vitesse, 0)  # Replace 0 with default value if needed
    carburant = categorical_encoders['carburant'].get(carburant, 0)  # Replace 0 with default value if needed

    # Fill missing values with mean
    cylindree = X_train['cylindree'].mean() if pd.isnull(cylindree) else cylindree
    kilometrage = X_train['kilometrage'].mean() if pd.isnull(kilometrage) else kilometrage
    nb_places = X_train['nb_places'].mean() if pd.isnull(nb_places) else nb_places
    nb_portes = X_train['nb_portes'].mean() if pd.isnull(nb_portes) else nb_portes
    nb_vitesses = X_train['nb_vitesses'].mean() if pd.isnull(nb_vitesses) else nb_vitesses
    puissance_fiscale = X_train['puissance_fiscale'].mean() if pd.isnull(puissance_fiscale) else puissance_fiscale
    puissance_physique = X_train['puissance_physique'].mean() if pd.isnull(puissance_physique) else puissance_physique
    annee = X_train['annee'].mean() if pd.isnull(annee) else annee

    # Predict car price
    prix_estime = rf_model.predict([[marque, cylindree, categorie, annee, boite_vitesse, carburant,
                                     kilometrage, nb_places, nb_portes, nb_vitesses, puissance_fiscale, puissance_physique]])

    return prix_estime

# Example usage:
#estimated_price = estimate_car_price('peugeot', 1199, 'berline', 2018, 'automatique', 'essence',
                                     #34857, 5, 5, 8, 7, 2018)
#print(estimated_price)

if __name__ == "__main__":
    # Récupérer les arguments de la ligne de commande
    args = sys.argv[1:]

    # Assurez-vous que le nombre correct d'arguments est fourni
    if len(args) != 12:
        print("Erreur: Nombre incorrect d'arguments.")
    else:
        # Appeler la fonction principale avec les arguments
        result = estimate_car_price(*args)

        # Afficher le résultat (ceci sera récupéré par VBA)
        print(result, end="")

