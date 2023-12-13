# prediction_model_functions.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

class CarPriceEstimator:
    """
    A class for estimating car prices using a RandomForestRegressor model.
    """

    def __init__(self):
        """
        Constructor for CarPriceEstimator class.
        Initializes categorical_encoders, quantitative_encoders, and rf_model attributes.
        """
        self.categorical_encoders = None
        self.quantitative_encoders = None
        self.rf_model = None

    def load_clean_data(self):
        """
        Load clean car data from a CSV file and return a DataFrame.

        Returns:
        pandas.DataFrame: The DataFrame containing the clean data for modeling.
        """
        script_directory = os.path.dirname(os.path.abspath(__file__))
        project_directory = os.path.dirname(os.path.dirname(script_directory))
        raw_data_path = os.path.join(project_directory, "data", "clean_data.csv")
        df = pd.read_csv(raw_data_path)
        return df

    def preprocess_data(self, data):
        """
        Preprocess the input data, encoding categorical variables and scaling quantitative variables.

        Parameters:
        - data (pandas.DataFrame): The input DataFrame containing car data.

        Returns:
        tuple: A tuple containing the preprocessed features (X) and the target variable (y).
        """
        X = data[["marque_et_modele", "cylindree", "categorie", "annee", "boite_vitesse", "carburant",
                  "kilometrage", "nb_places", "nb_portes", "nb_vitesses", "puissance_fiscale", "puissance_physique"]]
        y = data['prix']

        X.columns = [str(col) for col in X.columns]
        X_encoded = X.copy()

        self.categorical_encoders = {}
        self.quantitative_encoders = {}

        for column in X.columns:
            if X[column].dtype == "object":
                frequency = X[column].value_counts(normalize=True)
                X_encoded[column] = X[column].map(frequency)
                self.categorical_encoders[column] = frequency
            else:
                ss = StandardScaler()
                X_encoded[column] = ss.fit_transform(X[[column]])
                self.quantitative_encoders[column] = {'scaler': ss, 'mean': X[column].mean(), 'std': X[column].std()}

        return X_encoded, y

    def train_model(self, X, y):
        """
        Train a RandomForestRegressor model using the input features (X) and target variable (y).

        Parameters:
        - X (pandas.DataFrame): The input features.
        - y (pandas.Series): The target variable.

        Returns:
        None
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf_model = RandomForestRegressor(max_depth=15, min_samples_split=2, n_estimators=200)
        rf_model.fit(X_train, y_train)

        script_directory = os.path.dirname(os.path.abspath(__file__))

        # Set the directory name to save the model
        saved_model_directory = os.path.join(script_directory, 'saved model and encoders')

        #with open(os.path.join(saved_model_directory, 'rf_model.pkl'), 'wb') as file:
         #   pickle.dump(rf_model, file)

    def estimate_car_price(self, marque_et_modele, cylindree, categorie, annee, boite_vitesse, carburant,
                           kilometrage, nb_places, nb_portes, nb_vitesses, puissance_fiscale, puissance_physique):
        """
        Estimate the car price using the trained RandomForest model.

        Parameters:
        - marque_et_modele (str): The car make and model.
        - cylindree (float): The engine capacity.
        - categorie (str): The car category.
        - annee (float): The manufacturing year.
        - boite_vitesse (str): The gearbox type.
        - carburant (str): The fuel type.
        - kilometrage (float): The mileage of the car.
        - nb_places (float): The number of seats in the car.
        - nb_portes (float): The number of doors in the car.
        - nb_vitesses (float): The number of gears in the car.
        - puissance_fiscale (float): The fiscal horsepower of the car.
        - puissance_physique (float): The physical horsepower of the car.

        Returns:
        numpy.ndarray: The estimated car price.
        """
        # Get the script directory
        script_directory = os.path.dirname(os.path.abspath(__file__))
        # Set the directory name to save the model
        saved_model_directory = os.path.join(script_directory, 'saved model and encoders')
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'saved model and encoders/categorical_encoders.pkl'), 'rb') as file:
            self.categorical_encoders = pickle.load(file)
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'saved model and encoders/quantitative_encoders.pkl'), 'rb') as file:
            self.quantitative_encoders = pickle.load(file)
        with open(os.path.join(saved_model_directory, 'rf_model.pkl'), 'rb') as file:
            self.rf_model = pickle.load(file)

        marque_et_modele = self.categorical_encoders['marque_et_modele'].get(marque_et_modele, 0)
        categorie = self.categorical_encoders['categorie'].get(categorie, 0)
        boite_vitesse = self.categorical_encoders['boite_vitesse'].get(boite_vitesse, 0)
        carburant = self.categorical_encoders['carburant'].get(carburant, 0)

        cylindree = float(cylindree)
        kilometrage = float(kilometrage)
        nb_portes = float(nb_portes)
        nb_places = float(nb_places)
        nb_vitesses = float(nb_vitesses)
        puissance_fiscale = float(puissance_fiscale)
        puissance_physique = float(puissance_physique)
        annee = float(annee)
        cylindree = (cylindree - self.quantitative_encoders['cylindree']['mean']) / self.quantitative_encoders['cylindree']['std']
        kilometrage = (kilometrage - self.quantitative_encoders['kilometrage']['mean']) / self.quantitative_encoders['kilometrage']['std']
        nb_places = (nb_places - self.quantitative_encoders['nb_places']['mean']) / self.quantitative_encoders['nb_places']['std']
        nb_portes = (nb_portes - self.quantitative_encoders['nb_portes']['mean']) / self.quantitative_encoders['nb_portes']['std']
        nb_vitesses = (nb_vitesses - self.quantitative_encoders['nb_vitesses']['mean']) / self.quantitative_encoders['nb_vitesses']['std']
        puissance_fiscale = (puissance_fiscale - self.quantitative_encoders['puissance_fiscale']['mean']) / \
                            self.quantitative_encoders['puissance_fiscale']['std']
        puissance_physique = (puissance_physique - self.quantitative_encoders['puissance_physique']['mean']) / \
                             self.quantitative_encoders['puissance_physique']['std']
        annee = (annee - self.quantitative_encoders['annee']['mean']) / self.quantitative_encoders['annee']['std']

        estimated_price = self.rf_model.predict([[marque_et_modele, cylindree, categorie, annee, boite_vitesse, carburant,
                                                  kilometrage, nb_places, nb_portes, nb_vitesses, puissance_fiscale,
                                                  puissance_physique]])
        estimated_price = round(estimated_price[0])

        return estimated_price
