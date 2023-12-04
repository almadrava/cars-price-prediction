# Importation des bibliothèques et des données
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from numpy import mean
from numpy import std
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.graphics import tsaplots
from scipy.stats import shapiro
from scipy.stats import norm
from scipy.stats import ttest_ind
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

#importation des données et suppression des NA
data = pd.read_excel("clean_data.xlsx", sheet_name=0, header=0)
data = data.dropna()

#Conversion variable qualitative en  booléen
data = pd.get_dummies(data, columns=['boite_vitesse', 'categorie'])

# description de la base
print("\ndata.dtypes")
print(data.dtypes)
print("\ndata.head()")
print(data.head())

#Régression linéaire multiple (RLM)
X_numeric = data[['cylindree', 'kilometrage', 'nb_places', 'nb_portes', 'nb_vitesses',
                  'puissance_fiscale', 'puissance_physique', 'annee']]
X_bool = data[['boite_vitesse_manuelle', 'categorie_break', 'categorie_citadine', 'categorie_coupe-cabriolet', 'categorie_familiale', 'categorie_monospace', 'categorie_suv-4x4', 'categorie_utilitaire']].astype(int)
X_prix = data[['prix']]

X = pd.concat([X_numeric, X_bool], axis=1)
X = sm.add_constant(X)

y = data['prix']

modele_RLM = sm.OLS(y, X).fit()

# Attributs et résumé du modèle RLM
print("\nmodele_RLM.params")
print(modele_RLM.params)  # coefficients
print("\nmodele_RLM.summary")
print(modele_RLM.summary())

# Ordonner les variables explicatives par les valeurs des p-values
p_values = modele_RLM.pvalues[1:]  # Ignorer la p-value pour l'intercept
p_values.sort_values()

# Matrice de design, calcul de la matrice de corrélation et affichage sous forme de heatmap
XX = (pd.concat([X_prix, X_bool, X_numeric], axis=1))
cor_matrix = XX.corr()
print(cor_matrix)
sns.heatmap(cor_matrix, annot=True, cmap='coolwarm')
plt.show()

# Affichage des premières lignes de la matrice de design XX
print("\nmatrice de design XX")
print(XX.head())

# Régression linéaire multiple (RLM)
modele_RLM = sm.OLS(y, X).fit()

# Récupérer la matrice de conception (design matrix)
design_matrix = modele_RLM.model.exog

# Effectuer le test F en utilisant la fonction f_regression de scikit-learn
_, p_values = f_regression(design_matrix, y)

# Afficher les p-values
p_values_dict = dict(zip(X.columns, p_values))
sorted_p_values = sorted(p_values_dict.items(), key=lambda x: x[1])
print("\nP-values classées selon le test de Fisher:")
for name, p_value in sorted_p_values:
    print(f"{name}: {p_value}")

# Effectuer le test de Student pour chaque variable explicative
p_values_student = []
for column in X.columns:
    group1 = y[X[column] == 1]
    group0 = y[X[column] == 0]
    _, p_value = ttest_ind(group1, group0)
    p_values_student.append((column, p_value))

# Afficher les p-values du test de Student
print("\nP-values classées selon le test de Student:")
sorted_p_values_student = sorted(p_values_student, key=lambda x: x[1])
for name, p_value in sorted_p_values_student:
    print(f"{name}: {p_value}")

# On n'obtient pas le même classement, il est recommandé de retenir celui de Fisher

# Vérifier graphiquement la non-corrélation des erreurs
tsaplots.plot_acf(modele_RLM.resid, lags=40)
plt.title("Autocorrélations des erreurs")
plt.show()

# Tester la non-corrélation (d'ordre 1) des erreurs : test de Durbin-Watson
dw_statistic = durbin_watson(modele_RLM.resid)
print(f"Statistique de Durbin-Watson : {dw_statistic}")

# Vérifier l'hypothèse de linéarité entre la variable réponse et chaque variable explicative
# Graphiquement :
'''columns = X.columns[1:]  # Exclure la colonne constante
for col in columns:
    fig, ax = plt.subplots(figsize=(8, 6))
    sm.graphics.plot_partregress(endog=y, exog_i=X[col], exog_others=X.drop(col, axis=1), ax=ax, obs_labels=False)
    ax.set_title(f'{col} vs prix')
    plt.show()'''

# Régression linéaire avec transformation log(Prix)
model_log_price = sm.OLS(np.log(y), X).fit()

# Récupérer les résidus et les valeurs ajustées
residuals = model_log_price.resid
fitted_values = model_log_price.fittedvalues

# Créer le graphique des résidus par rapport aux valeurs ajustées
plt.figure(figsize=(8, 6))
plt.scatter(fitted_values, residuals, alpha=0.7)
plt.title('Graphique des résidus par rapport aux valeurs ajustées')
plt.xlabel('Valeurs ajustées (Fitted values)')
plt.ylabel('Résidus')
plt.show()

# Vérifier l'hypothèse d'homoscedasticité des erreurs graphiquement
sm.graphics.plot_fit(modele_RLM, 0)
plt.title("Vérification de l'homoscedasticité des erreurs")
plt.show()

# Test d'homoscedasticité de Breusch-Pagan
_, bp_p_value, _, _ = het_breuschpagan(modele_RLM.resid, modele_RLM.model.exog)
print(f"p-value du test de Breusch-Pagan : {bp_p_value}")

#######Normal Q-Q plot##########

#### Graphiquement : normal Q-Q plot 1
sm.qqplot(modele_RLM.resid, line='s')
plt.title("Normal Q-Q Plot (Toutes les variables explicatives)")
plt.show()

#### Graphiquement : normal Q-Q plot 2
#Régression linéaire multiple (RLM)
X_numeric = data[['puissance_physique', 'nb_vitesses', 'annee']]
X_bool = data[['categorie_coupe-cabriolet']].astype(int)

X = pd.concat([X_numeric,X_bool], axis=1)
X = sm.add_constant(X)

modele_RLM = sm.OLS(y, X).fit()

# Graphiquement : normal Q-Q plot 2
sm.qqplot(modele_RLM.resid, line='s')
plt.title("Normal Q-Q Plot (4 variables les plus corrélées à la variable refférence)")
plt.show()

##### Graphiquement : normal Q-Q plot 3
#Régression linéaire multiple (RLM)
X_numeric = data[['cylindree', 'kilometrage', 'nb_places', 'nb_portes', 'puissance_physique', 'annee']]
X_bool = data[['boite_vitesse_manuelle', 'categorie_citadine', 'categorie_familiale', 'categorie_monospace', 'categorie_suv-4x4', 'categorie_utilitaire']].astype(int)

X = pd.concat([X_numeric, X_bool], axis=1)
X = sm.add_constant(X)

modele_RLM = sm.OLS(y, X).fit()

# Graphiquement : normal Q-Q plot 3
sm.qqplot(modele_RLM.resid, line='s')
plt.title("Normal Q-Q Plot (Pvalue < 400)")
plt.show()
##################################################

# Histogramme versus densité normale
residus = modele_RLM.resid
plt.hist(residus, bins='auto', density=True, alpha=0.7, color='blue', edgecolor='black')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean(residus), std(residus))
plt.plot(x, p, 'k', linewidth=2)
title = "Histogramme des résidus et densité normale"
plt.title(title)
plt.show()

# Test de Shapiro-Wilk pour tester l'hypothèse de normalité du terme d'erreur
shapiro_stat, shapiro_p_value = shapiro(modele_RLM.resid)
print(f"Statistique de test de Shapiro-Wilk : {shapiro_stat}")
print(f"P-value du test de Shapiro-Wilk : {shapiro_p_value}")

################AIC-BIC_model########################

# Variables numériques
X_numeric = data[['cylindree', 'kilometrage', 'nb_places', 'nb_portes', 'nb_vitesses',
                  'puissance_fiscale', 'puissance_physique', 'annee']]

# Variables booléennes
X_bool = data[['boite_vitesse_manuelle', 'categorie_break', 'categorie_citadine', 'categorie_coupe-cabriolet',
               'categorie_familiale', 'categorie_monospace', 'categorie_suv-4x4', 'categorie_utilitaire']].astype(int)


# Matrice de design sans l'intercept
X = pd.concat([X_numeric, X_bool], axis=1)
X = sm.add_constant(X)


# Sélection de modèle avec AIC
best_aic = float('inf')
best_aic_model = None
best_aic_n_variables = None
for i in range(2, X.shape[1] + 1):
    model = LinearRegression()
    scores = cross_val_score(model, X, y, cv=i, scoring='neg_mean_squared_error')
    aic = 2 * i - 2 * np.sum(scores)
    if aic < best_aic:
        best_aic = aic
        best_aic_model = model
        best_aic_n_variables = i

# Ajuster le modèle optimal sur l'ensemble complet des données
best_aic_model.fit(X, y)

# Afficher les coefficients
print("\nModèle optimal avec AIC (Nombre optimal de variables =", best_aic_n_variables, "):")
print("\nCoefficients:", best_aic_model.coef_)
print("\nIntercept:", best_aic_model.intercept_)

# Sélection de modèle avec BIC
best_bic = float('inf')
best_bic_model = None
best_bic_n_variables = None
for i in range(2, X.shape[1] + 1):
    model = LinearRegression()
    scores = cross_val_score(model, X, y, cv=i, scoring='neg_mean_squared_error')
    bic = i * np.log(X.shape[0]) - 2 * np.sum(scores)
    if bic < best_bic:
        best_bic = bic
        best_bic_model = model
        best_bic_n_variables = i

# Ajuster le modèle optimal sur l'ensemble complet des données
best_bic_model.fit(X, y)

# Afficher les coefficients
print("\nModèle optimal avec BIC (Nombre optimal de variables =", best_bic_n_variables, "):")
print("\nCoefficients:", best_bic_model.coef_)
print("\nIntercept:", best_bic_model.intercept_)

# Noms de colonnes
column_names = X.columns

# Affichage des noms de colonnes et de leurs coefficients pour le modèle AIC
print("\nModèle optimal avec AIC:")
for name, coef in zip(column_names, best_aic_model.coef_):
    print(f"{name}: {coef}")

# Affichage des noms de colonnes et de leurs coefficients pour le modèle BIC
print("\nModèle optimal avec BIC:")
for name, coef in zip(column_names, best_bic_model.coef_):
    print(f"{name}: {coef}")

# Affichage des noms de variables pour le modèle optimal avec AIC
selected_variables_aic = X.columns[best_aic_model.coef_ != 0]
print("\nVariables sélectionnées par AIC:", selected_variables_aic)

# Affichage des noms de variables pour le modèle optimal avec BIC
selected_variables_bic = X.columns[best_bic_model.coef_ != 0]
print("\nVariables sélectionnées par BIC:", selected_variables_bic)

################Ridge_model########################

# Régression ridge avec cross-validation pour trouver le meilleur alpha (équivalent à lambda en glmnet)
parameters = {'alpha': np.logspace(-6, 6, 13)}
reg_ridge = Ridge()
reg_cv_ridge = GridSearchCV(reg_ridge, parameters, cv=10, scoring='neg_mean_squared_error')
reg_cv_ridge.fit(StandardScaler().fit_transform(X), np.log(data['prix']))

# Plot du chemin de régularisation
alphas = np.logspace(-6, 6, 13)
plt.figure(figsize=(8, 6))
plt.semilogx(alphas, reg_cv_ridge.cv_results_['mean_test_score'])
plt.xlabel('Alpha')
plt.ylabel('Negative Mean Squared Error')
plt.title('Chemin de régularisation (Régression Ridge)')
plt.show()

# Meilleur alpha
best_alpha = reg_cv_ridge.best_params_['alpha']
print(f"Meilleur alpha : {best_alpha}")

# Coefficients avec le meilleur alpha
reg_ridge_best_alpha = Ridge(alpha=best_alpha)
reg_ridge_best_alpha.fit(StandardScaler().fit_transform(X), np.log(data['prix']))

# Variables sélectionnées (non nulles)
ridge_coef = reg_ridge_best_alpha.coef_
selected_variables = np.where(ridge_coef != 0)[0]
selected_variable_names = X.columns[selected_variables]
print("\nVariables sélectionnées (Ridge) :")
print(selected_variable_names)

# Erreur de prévision du modèle ridge optimal
erreur_modele_ridge_opt = -reg_cv_ridge.best_score_
print(f"Erreur de prévision du modèle ridge optimal : {erreur_modele_ridge_opt}")

# Erreur de prévision du modèle RLM complet
modele_RLM_complet = Ridge(alpha=0)  # Alpha=0 correspond à la régression linéaire sans pénalité
erreur_modele_RLM_complet = -cross_val_score(modele_RLM_complet, StandardScaler().fit_transform(X),
                                             np.log(data['prix']), scoring='neg_mean_squared_error', cv=10).mean()
print(f"Erreur de prévision du modèle RLM complet : {erreur_modele_RLM_complet}")

################Lasso_model########################

# Régression Lasso avec cross-validation pour trouver le meilleur alpha (équivalent à lambda en glmnet)
lasso_cv = LassoCV(cv=10, alphas=np.logspace(-6, 6, 13))
lasso_cv.fit(StandardScaler().fit_transform(X), np.log(data['prix']))

# Plot du chemin de régularisation
alphas_lasso = np.logspace(-6, 6, 13)
plt.figure(figsize=(8, 6))
plt.semilogx(alphas_lasso, lasso_cv.mse_path_.mean(axis=1))
plt.xlabel('Alpha')
plt.ylabel('Mean Squared Error')
plt.title('Chemin de régularisation (Régression Lasso)')
plt.show()

# Meilleur alpha
best_alpha_lasso = lasso_cv.alpha_
print(f"Meilleur alpha (Lasso) : {best_alpha_lasso}")

# Coefficients avec le meilleur alpha
lasso_coef = lasso_cv.coef_
selected_variables_lasso = np.where(lasso_coef != 0)[0]
selected_variable_names_lasso = X.columns[selected_variables_lasso]
print("\nVariables sélectionnées (Lasso) :")
print(selected_variable_names_lasso)

# Erreur de prévision du modèle lasso optimal
erreur_modele_lasso_opt = lasso_cv.mse_path_.mean(axis=1).min()
print(f"Erreur de prévision du modèle lasso optimal : {erreur_modele_lasso_opt}")

# Erreur de prévision du modèle RLM complet
modele_RLM_complet = LinearRegression()
erreur_modele_RLM_complet = -cross_val_score(modele_RLM_complet, StandardScaler().fit_transform(X),
                                             np.log(data['prix']), scoring='neg_mean_squared_error', cv=10).mean()

print(f"Erreur de prévision du modèle RLM complet : {erreur_modele_RLM_complet}")

################RandomForest_Model########################

# Régression avec Random Forest
random_forest = RandomForestRegressor(n_estimators=100, random_state=42)  # Ajuster les hyperparamètres ici

random_forest.fit(X, np.log(data['prix']))

# Prédictions
predictions_rf = random_forest.predict(X)

# Visualisation de l'importance des fonctionnalités
feature_importances = random_forest.feature_importances_

# Création d'un DataFrame pour faciliter la visualisation
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

# Tri par ordre décroissant d'importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Sélection des variables importantes (seuil arbitraire)
threshold = 0.02  # Ajustez ce seuil selon vos besoins
selected_variables_rf = importance_df[importance_df['Importance'] > threshold]['Feature'].values

# Affichage des variables sélectionnées
print("\nVariables sélectionnées (Random Forest) :")
print(selected_variables_rf)

# Tracer la barre d'importance avec hue
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', hue='Feature', data=importance_df, palette='viridis', legend=False)
plt.title('Importance des fonctionnalités (Random Forest)')
plt.show()


# Erreur de prévision du modèle Random Forest
erreur_modele_rf = -cross_val_score(random_forest, XX, np.log(data['prix']), scoring='neg_mean_squared_error', cv=10).mean()

print(f"Erreur de prévision du modèle Random Forest : {erreur_modele_rf}")