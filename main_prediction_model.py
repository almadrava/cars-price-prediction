"""
Main script for estimating car prices using the RandomForest model.
"""
import sys
from src.tools.model_prediction.prediction_model_functions import CarPriceEstimator

def main():
    """
    Main function for script execution.
    Loads data, preprocesses, trains the model, and estimates car prices based on command-line arguments.
    Prints the result.
    """
    car_price_estimator = CarPriceEstimator()
    data = car_price_estimator.load_clean_data()
    X_encoded, y = car_price_estimator.preprocess_data(data)
    car_price_estimator.train_model(X_encoded, y)

    args = sys.argv[1:]

    if len(args) != 12:
        print("Erreur: Nombre incorrect d'arguments.")
    else:
        result = car_price_estimator.estimate_car_price(*args)
        print(result, end="")

if __name__ == "__main__":
    main()
