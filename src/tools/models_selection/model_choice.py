#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[3]:


#importation des données et suppression des NA
data = pd.read_excel("D:/clean_data.xlsx", sheet_name=0, header=0)
data = data.dropna()

#Conversion variable qualitative en  booléen
data = pd.get_dummies(data, columns=['boite_vitesse', 'categorie'])

# description de la base
print("\ndata.dtypes")
print(data.dtypes)
print("\ndata.head()")
print(data.head())


# In[4]:


#Régression linéaire multiple (RLM)
X_numeric = data[['cylindree', 'kilometrage', 'nb_places', 'nb_portes', 'nb_vitesses',
                  'puissance_fiscale', 'puissance_physique', 'annee']]
X_bool = data[['boite_vitesse_manuelle', 'categorie_break', 'categorie_citadine', 'categorie_coupe-cabriolet', 'categorie_familiale', 'categorie_monospace', 'categorie_suv-4x4', 'categorie_utilitaire']].astype(int)
X_prix = data[['prix']]

X = pd.concat([X_numeric, X_bool], axis=1)
X = sm.add_constant(X)

y = data['prix']


# In[5]:


modele_RLM = sm.OLS(y, X).fit()


# In[6]:


# Attributs et résumé du modèle RLM
print("\nmodele_RLM.params")
print(modele_RLM.params)  # coefficients
print("\nmodele_RLM.summary")
print(modele_RLM.summary())


# In[7]:


# Ordonner les variables explicatives par les valeurs des p-values
p_values = modele_RLM.pvalues[1:]  # Ignorer la p-value pour l'intercept
p_values.sort_values()


# In[8]:


# Matrice de design, calcul de la matrice de corrélation et affichage sous forme de heatmap
XX = (pd.concat([X_prix, X_bool, X_numeric], axis=1))
cor_matrix = XX.corr()
print(cor_matrix)
sns.heatmap(cor_matrix, annot=True, cmap='coolwarm')
plt.show()

# Affichage des premières lignes de la matrice de design XX
print("\nmatrice de design XX")
print(XX.head())


# In[9]:


# Récupérer la matrice de conception (design matrix)
design_matrix = modele_RLM.model.exog


# In[10]:


# Effectuer le test F en utilisant la fonction f_regression de scikit-learn
_, p_values = f_regression(design_matrix, y)


# In[11]:


# Afficher les p-values
p_values_dict = dict(zip(X.columns, p_values))
sorted_p_values = sorted(p_values_dict.items(), key=lambda x: x[1])
print("\nP-values classées selon le test de Fisher:")
for name, p_value in sorted_p_values:
    print(f"{name}: {p_value}")


# In[12]:


# Effectuer le test de Student pour chaque variable explicative
p_values_student = []
for column in X.columns:
    group1 = y[X[column] == 1]
    group0 = y[X[column] == 0]
    _, p_value = ttest_ind(group1, group0)
    p_values_student.append((column, p_value))


# In[13]:


# Afficher les p-values du test de Student
print("\nP-values classées selon le test de Student:")
sorted_p_values_student = sorted(p_values_student, key=lambda x: x[1])
for name, p_value in sorted_p_values_student:
    print(f"{name}: {p_value}")

# On n'obtient pas le même classement, il est recommandé de retenir celui de Fisher


# In[14]:


# Vérifier graphiquement la non-corrélation des erreurs
tsaplots.plot_acf(modele_RLM.resid, lags=40)
plt.title("Autocorrélations des erreurs")
plt.show()


# In[15]:


# Tester la non-corrélation (d'ordre 1) des erreurs : test de Durbin-Watson
dw_statistic = durbin_watson(modele_RLM.resid)
print(f"Statistique de Durbin-Watson : {dw_statistic}")


# In[16]:


# Régression linéaire avec transformation log(Prix)
model_log_price = sm.OLS(np.log(y), X).fit()

# Récupérer les résidus et les valeurs ajustées
residuals = model_log_price.resid
fitted_values = model_log_price.fittedvalues


# In[17]:


# Créer le graphique des résidus par rapport aux valeurs ajustées
plt.figure(figsize=(8, 6))
plt.scatter(fitted_values, residuals, alpha=0.7)
plt.title('Graphique des résidus par rapport aux valeurs ajustées')
plt.xlabel('Valeurs ajustées (Fitted values)')
plt.ylabel('Résidus')
plt.show()


# In[18]:


# Vérifier l'hypothèse d'homoscedasticité des erreurs graphiquement
sm.graphics.plot_fit(modele_RLM, 0)
plt.title("Vérification de l'homoscedasticité des erreurs")
plt.show()


# In[19]:


# Test d'homoscedasticité de Breusch-Pagan
_, bp_p_value, _, _ = het_breuschpagan(modele_RLM.resid, modele_RLM.model.exog)
print(f"p-value du test de Breusch-Pagan : {bp_p_value}")


# In[20]:


#######Normal Q-Q plot##########

#### Graphiquement : normal Q-Q plot 1
sm.qqplot(modele_RLM.resid, line='s')
plt.title("Normal Q-Q Plot (Toutes les variables explicatives)")
plt.show()


# In[21]:


### Graphiquement : normal Q-Q plot 2
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


# In[22]:


#### Graphiquement : normal Q-Q plot 3
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


# In[23]:


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


# In[24]:


# Test de Shapiro-Wilk pour tester l'hypothèse de normalité du terme d'erreur
shapiro_stat, shapiro_p_value = shapiro(modele_RLM.resid)
print(f"Statistique de test de Shapiro-Wilk : {shapiro_stat}")
print(f"P-value du test de Shapiro-Wilk : {shapiro_p_value}")


# In[25]:


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


# In[42]:


# Faire des prédictions sur les données d'entraînement
predictions = lasso_cv.predict(StandardScaler().fit_transform(X))

# Calculer le coefficient de détermination R²
r2 = r2_score(np.log(data['prix']), predictions)

# Afficher le résultat
print("Le coefficient de détermination R² est :", r2)


# In[26]:


# Meilleur alpha
best_alpha_lasso = lasso_cv.alpha_
print(f"Meilleur alpha (Lasso) : {best_alpha_lasso}")

# Coefficients avec le meilleur alpha
lasso_coef = lasso_cv.coef_
selected_variables_lasso = np.where(lasso_coef != 0)[0]
selected_variable_names_lasso = X.columns[selected_variables_lasso]
print("\nVariables sélectionnées (Lasso) :")
print(selected_variable_names_lasso)


# In[27]:


# Erreur de prévision du modèle lasso optimal
erreur_modele_lasso_opt = lasso_cv.mse_path_.mean(axis=1).min()
print(f"Erreur de prévision du modèle lasso optimal : {erreur_modele_lasso_opt}")

# Erreur de prévision du modèle RLM complet
modele_RLM_complet = LinearRegression()
erreur_modele_RLM_complet = -cross_val_score(modele_RLM_complet, StandardScaler().fit_transform(X),
                                             np.log(data['prix']), scoring='neg_mean_squared_error', cv=10).mean()

print(f"Erreur de prévision du modèle RLM complet : {erreur_modele_RLM_complet}")


# In[29]:


################RandomForest_Model########################

# Régression avec Random Forest
random_forest = RandomForestRegressor(n_estimators=100, random_state=42)  # Ajuster les hyperparamètres ici

random_forest.fit(X, np.log(data['prix']))

# Prédictions
predictions_rf = random_forest.predict(X)



# In[30]:


# Visualisation de l'importance des fonctionnalités
feature_importances = random_forest.feature_importances_

# Création d'un DataFrame pour faciliter la visualisation
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})


# In[31]:


# Tri par ordre décroissant d'importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Sélection des variables importantes (seuil arbitraire)
threshold = 0.02  # Ajustez ce seuil selon vos besoins
selected_variables_rf = importance_df[importance_df['Importance'] > threshold]['Feature'].values


# In[32]:


# Affichage des variables sélectionnées
print("\nVariables sélectionnées (Random Forest) :")
print(selected_variables_rf)


# In[34]:


# Tracer la barre d'importance avec hue
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', hue='Feature', data=importance_df, palette='viridis')
plt.title('Importance des fonctionnalités (Random Forest)')
plt.show()


# In[35]:


# Erreur de prévision du modèle Random Forest
erreur_modele_rf = -cross_val_score(random_forest, XX, np.log(data['prix']), scoring='neg_mean_squared_error', cv=10).mean()

print(f"Erreur de prévision du modèle Random Forest : {erreur_modele_rf}")


# In[41]:


from sklearn.metrics import r2_score

# Calculer le coefficient de détermination R²
r2 = r2_score(np.log(data['prix']), predictions_rf)

# Afficher le résultat
print("Le coefficient de détermination R² est :", r2)


# In[43]:


# Le modele choisi est donc le RandomForest car meilleur R² et meilleure erreur de prevision 
# Calculer les résidus
residuals = np.log(data['prix']) - predictions_rf

# Graphique QQ Normalite
sm.qqplot(residuals, line='s')
plt.show()


# In[45]:


from statsmodels.graphics.tsaplots import plot_acf

# Fonction d'autocorrélation
plot_acf(residuals, lags=20)
plt.show()


# In[46]:


# Graphique de résidus vs. prédictions autocorrelation individu 
plt.scatter(predictions_rf, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Prédictions')
plt.ylabel('Résidus')
plt.show()


# In[47]:


# Graphique de résidus vs. prédictions homoscedasticite
plt.scatter(predictions_rf, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Prédictions')
plt.ylabel('Résidus')
plt.show()


# In[48]:


# Test de normalité
_, p_value = shapiro(residuals)
print("Test de normalité (Shapiro-Wilk) p-value :", p_value)


# In[49]:


# Test de Durbin-Watson
dw_statistic = durbin_watson(residuals)
print("Test de Durbin-Watson :", dw_statistic)


# In[54]:


################### Test #########################


# In[55]:


from scipy.stats import anderson

# Calculer les résidus
residuals = np.log(data['prix']) - predictions_rf

# Test d'Anderson-Darling
result = anderson(residuals)
print("Statistic d'Anderson-Darling :", result.statistic)
print("P-value :", result.critical_values)


# In[56]:


from scipy.stats import boxcox

# Appliquer la transformation de Box-Cox à la variable dépendante
transformed_prices, lambda_value = boxcox(data['prix'])

# Utiliser le modèle de régression avec la variable dépendante transformée
random_forest.fit(X, np.log(transformed_prices))


# In[57]:


# Prédictions
predictions_test = random_forest.predict(X)
residualstest = np.log(data['prix']) - predictions_test


# In[58]:


# Test d'Anderson-Darling
result = anderson(residualstest)
print("Statistic d'Anderson-Darling :", result.statistic)
print("P-value :", result.critical_values)


# In[59]:


# Calculer les résidus
residuals = np.log(data['prix']) - predictions_rf

# Calculer les résidus standardisés
standardized_residuals = residuals / np.std(residuals)


# In[60]:


# Identifier les indices des valeurs aberrantes
outlier_indices = np.where(np.abs(standardized_residuals) > 3)[0]

# Supprimer les valeurs aberrantes
X_no_outliers = np.delete(X, outlier_indices, axis=0)
y_no_outliers = np.delete(np.log(data['prix']), outlier_indices)


# In[61]:


# Créer une nouvelle instance de RandomForestRegressor
random_forest_no_outliers = RandomForestRegressor(n_estimators=100, random_state=42)

# Ajuster le modèle sur les données sans les valeurs aberrantes
random_forest_no_outliers.fit(X_no_outliers, y_no_outliers)


# In[62]:


# Prédictions sans les valeurs aberrantes
predictions_rf_no_outliers = random_forest_no_outliers.predict(X_no_outliers)

# Calculer les nouveaux résidus
residuals_no_outliers = y_no_outliers - predictions_rf_no_outliers


# In[63]:


# Graphique QQ Normalite
sm.qqplot(residuals_no_outliers, line='s')
plt.show()


# In[64]:


# Fonction d'autocorrélation
plot_acf(residuals_no_outliers, lags=20)
plt.show()


# In[66]:


# Graphique de résidus vs. prédictions autocorrelation individu 
plt.scatter(predictions_rf_no_outliers, residuals_no_outliers)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Prédictions')
plt.ylabel('Résidus')
plt.show()


# In[67]:


# Test de normalité
_, p_value = shapiro(residuals_no_outliers)
print("Test de normalité (Shapiro-Wilk) p-value :", p_value)


# In[68]:


# Test d'Anderson-Darling
result = anderson(residuals_no_outliers)
print("Statistic d'Anderson-Darling :", result.statistic)
print("P-value :", result.critical_values)


# In[69]:


X


# In[72]:


# Appliquer la transformation de Box-Cox à la variable dépendante
transformed_prices, lambda_value = boxcox(y_no_outliers)



# In[73]:


# Utiliser le modèle de régression avec la variable dépendante transformée
random_forest.fit(X_no_outliers, transformed_prices)


# In[74]:


# Prédictions
predictions_test = random_forest.predict(X_no_outliers)
residualstest = transformed_prices - predictions_test


# In[75]:


# Test de normalité
_, p_value = shapiro(residualstest)
print("Test de normalité (Shapiro-Wilk) p-value :", p_value)


# In[76]:


# Test d'Anderson-Darling
result = anderson(residualstest)
print("Statistic d'Anderson-Darling :", result.statistic)
print("P-value :", result.critical_values)


# In[77]:


######### Resultats final ######


# In[ ]:


######### Modele final Random Forest sans les outliers ######


# In[78]:


feature_importances = random_forest_no_outliers.feature_importances_

# Afficher l'importance des variables
for i, importance in enumerate(feature_importances):
    print(f"Variable {i+1}: Importance = {importance}")


# In[79]:


X_no_outliers


# In[80]:


df_X_no_outliers = pd.DataFrame(X_no_outliers)


# In[81]:


df_X_no_outliers


# In[82]:


X


# In[ ]:




