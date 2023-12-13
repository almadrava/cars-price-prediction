# Importing libraries
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from src.tools.model_prediction.prediction_model_functions import CarPriceEstimator

# Load and clean data
data1 = CarPriceEstimator.load_clean_data()

# Correlation matrix and heatmap for quantitative variables
X_quanti = data1.select_dtypes(include=['int64', 'float64'])
correlation_matrix = X_quanti.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

# Feature selection and conversion of qualitative variables to boolean
X = data1.drop(['prix', 'marque', 'modele', 'marque_et_modele', 'couleur', 'nb_roues_motrices'], axis=1)
data = pd.get_dummies(data1, columns=['boite_vitesse', 'categorie', 'critair', 'carburant'])

# Display data types and head of the dataset
print("\ndata.dtypes")
print(data.dtypes)
print("\ndata.head()")
print(data.head())

# Split features into numeric and boolean
X_numeric = data[['cylindree', 'kilometrage', 'nb_places', 'nb_portes', 'nb_vitesses',
                  'puissance_fiscale', 'puissance_physique', 'annee']]
X_bool = data[['boite_vitesse_manuelle', 'categorie_break', 'categorie_citadine', 'categorie_coupe-cabriolet',
               'categorie_familiale', 'categorie_monospace', 'categorie_suv-4x4', 'categorie_utilitaire',
               'critair_niveau 0', 'critair_niveau 1', 'critair_niveau 2', 'carburant_diesel', 'carburant_electrique',
               'carburant_essence', 'carburant_hybride']].astype(int)
X_prix = data[['prix']]

# Concatenate numeric and boolean features
X = pd.concat([X_numeric, X_bool], axis=1)

# Correlation matrix for the selected features
correlation_matrix = X.corr()

# Heatmap for the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

# Display column names
print(data1.columns)

# High correlation matrix (> 0.7)
high_corr_matrix = correlation_matrix[correlation_matrix.abs() > 0.7]

# Heatmap for high correlation matrix
sns.heatmap(high_corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('High Correlation Matrix (coefficients > 0.7)')
plt.show()

