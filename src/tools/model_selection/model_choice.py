# Importing necessary librairies
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
from scipy.stats import anderson,boxcox,shapiro,norm,ttest_ind
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import r2_score
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
#from src.tools.model_prediction.prediction_model_functions import CarPriceEstimator

# Load and clean data
#data = CarPriceEstimator.load_clean_data()
data = pd.read_excel("clean_data.xlsx", sheet_name=0, header=0)
data = data.dropna()

# Convert qualitative variable to numerical using impact encoding
def impact_encode(data, column, target):
    encoding_map = data.groupby(column)[target].mean().to_dict()
    data[column + '_encoded'] = data[column].map(encoding_map)
    data = data.drop(column, axis=1)
    return data

# List of qualitative variables to encode
qualitative_variables = ['marque_et_modele', 'marque', 'couleur', 'critair', 'boite_vitesse', 'categorie', 'nb_roues_motrices']  # 'marque_et_modele' column <=> 'modele' column

# Apply 'impact encoding' to each qualitative variable
for var in qualitative_variables:
    data = impact_encode(data, var, 'prix')


# Display basic information about the data
print("\ndata.dtypes")
print(data.dtypes)
print("\ndata.head()")
print(data.head())

##########################################################
############ Multiple linear regression model ############
##########################################################

# Perform Multiple Linear Regression (MLR)
X_numeric = data[['cylindree', 'kilometrage', 'nb_places', 'nb_portes', 'nb_vitesses',
                  'puissance_fiscale', 'puissance_physique', 'annee']]
X_encoded = data[['marque_et_modele_encoded', 'marque_encoded','couleur_encoded','critair_encoded',
                  'boite_vitesse_encoded', 'categorie_encoded','nb_roues_motrices_encoded']]
X_prix = data[['prix']]

X = pd.concat([X_numeric, X_encoded], axis=1)
X = sm.add_constant(X)

y = data['prix']

modele_RLM = sm.OLS(y, X).fit()

# Display attributes and summary of the model
print("\nmodele_RLM.params")
print(modele_RLM.params)  # coefficients
print("\nmodele_RLM.summary")
print(modele_RLM.summary())

# Order explanatory variables by p-values
p_values = modele_RLM.pvalues[1:]  # Ignore the p-value for the intercept
p_values.sort_values()

# Design matrix, correlation matrix calculation, and heatmap display
XX = (pd.concat([X_prix, X_encoded, X_numeric], axis=1))
cor_matrix = XX.corr()
print(cor_matrix)
sns.heatmap(cor_matrix, annot=True, cmap='coolwarm')
plt.show()

# Display the first rows of the design matrix XX
print("\nmatrice de design XX")
print(XX.head())

# Perform the F-test using the f_regression function from scikit-learn
design_matrix = modele_RLM.model.exog
_, p_values_fisher = f_regression(design_matrix, y)

# Display p-values
p_values_dict = dict(zip(X.columns, p_values_fisher))
sorted_p_values_fisher = sorted(p_values_dict.items(), key=lambda x: x[1])
print("\nP-values classées selon le test de Fisher:")
for name, p_value in sorted_p_values_fisher:
    print(f"{name}: {p_value}")

# Perform Student's t-test for each explanatory variable
p_values_student = []
for column in X.columns:
    group1 = y[X[column] == 1]
    group0 = y[X[column] == 0]
    _, p_value = ttest_ind(group1, group0)
    p_values_student.append((column, p_value))

# Display p-values of Student's t-test
print("\nP-values classées selon le test de Student:")
sorted_p_values_student = sorted(p_values_student, key=lambda x: x[1])
for name, p_value in sorted_p_values_student:
    print(f"{name}: {p_value}")

# Check for autocorrelation of residuals graphically
tsaplots.plot_acf(modele_RLM.resid, lags=40)
plt.title("Autocorrélations des erreurs")
plt.show()

# Test for autocorrelation (order 1) of residuals: Durbin-Watson test
dw_statistic = durbin_watson(modele_RLM.resid)
print(f"Statistique de Durbin-Watson : {dw_statistic}")

# Linear regression with log-transformed price
model_log_price = sm.OLS(np.log(y), X).fit()

## Retrieve residuals and fitted values
residuals = model_log_price.resid
fitted_values = model_log_price.fittedvalues

## Create a scatter plot of residuals against fitted values
plt.figure(figsize=(8, 6))
plt.scatter(fitted_values, residuals, alpha=0.7)
plt.title('Residuals vs Fitted Values Plot')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()

## Breusch-Pagan test for homoscedasticity
_, bp_p_value, _, _ = het_breuschpagan(model_log_price.resid, model_log_price.model.exog)
print(f"P-value from Breusch-Pagan test: {bp_p_value}")


# Normal Q-Q plots for residuals with different variable sets

## Q-Q plot 1: All explanatory variables
sm.qqplot(modele_RLM.resid, line='s')
plt.title("Normal Q-Q Plot (All Explanatory Variables)")
plt.show()

# Histogram vs. normal density
residuals = modele_RLM.resid
plt.hist(residuals, bins='auto', density=True, alpha=0.7, color='blue', edgecolor='black')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean(residuals), std(residuals))
plt.plot(x, p, 'k', linewidth=2)
title = "Histogram of Residuals and Normal Density"
plt.title(title)
plt.show()

# Shapiro-Wilk test for normality of residuals
shapiro_stat, shapiro_p_value = shapiro(modele_RLM.resid)
print(f"Shapiro-Wilk Test Statistic: {shapiro_stat}")
print(f"P-value from Shapiro-Wilk Test: {shapiro_p_value}")

#######################################################################
############ AIC-BIC model selection for linear regression ############
#######################################################################

# Model selection with AIC
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

# Fit the optimal AIC model on the complete data
best_aic_model.fit(X, y)

# Display coefficients
print("\nOptimal model with AIC (Optimal number of variables =", best_aic_n_variables, "):")
print("\nCoefficients:", best_aic_model.coef_)
print("\nIntercept:", best_aic_model.intercept_)

# Model selection with BIC
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

# Fit the optimal BIC model on the complete data
best_bic_model.fit(X, y)

# Display coefficients
print("\nOptimal model with BIC (Optimal number of variables =", best_bic_n_variables, "):")
print("\nCoefficients:", best_bic_model.coef_)
print("\nIntercept:", best_bic_model.intercept_)

# Column names
column_names = X.columns

# Display column names and their coefficients for the AIC model
print("\nOptimal model with AIC:")
for name, coef in zip(column_names, best_aic_model.coef_):
    print(f"{name}: {coef}")

# Display column names and their coefficients for the BIC model
print("\nOptimal model with BIC:")
for name, coef in zip(column_names, best_bic_model.coef_):
    print(f"{name}: {coef}")

# Display variable names for the optimal AIC model
selected_variables_aic = X.columns[best_aic_model.coef_ != 0]
print("\nVariables selected by AIC:", selected_variables_aic)

# Display variable names for the optimal BIC model
selected_variables_bic = X.columns[best_bic_model.coef_ != 0]
print("\nVariables selected by BIC:", selected_variables_bic)

#####################################################
############ Ridge-Lasso model selection ############
#####################################################

# Ridge regression with cross-validation for finding the best alpha

# Define the range of alpha values
alphas = np.logspace(-6, 6, 13)

# Create Ridge regression model
reg_ridge = Ridge()

# Perform grid search with cross-validation
parameters = {'alpha': alphas}
reg_cv_ridge = GridSearchCV(reg_ridge, parameters, cv=10, scoring='neg_mean_squared_error')
reg_cv_ridge.fit(StandardScaler().fit_transform(X), np.log(data['prix']))

# Plot the regularization path
plt.figure(figsize=(8, 6))
plt.semilogx(alphas, reg_cv_ridge.cv_results_['mean_test_score'])
plt.xlabel('Alpha')
plt.ylabel('Negative Mean Squared Error')
plt.title('Regularization Path (Ridge Regression)')
plt.show()

# Best alpha
best_alpha = reg_cv_ridge.best_params_['alpha']
print(f"Best alpha: {best_alpha}")

# Coefficients with the best alpha
reg_ridge_best_alpha = Ridge(alpha=best_alpha)
reg_ridge_best_alpha.fit(StandardScaler().fit_transform(X), np.log(data['prix']))

# Selected variables (non-zero coefficients)
ridge_coef = reg_ridge_best_alpha.coef_
selected_variables = np.where(ridge_coef != 0)[0]
selected_variable_names = X.columns[selected_variables]
print("\nSelected variables (Ridge):")
print(selected_variable_names)

# Prediction error of the optimal ridge model
erreur_modele_ridge_opt = -reg_cv_ridge.best_score_
print(f"Prediction error of the optimal ridge model: {erreur_modele_ridge_opt}")

# Prediction error of the complete RLM model
modele_RLM_complet = Ridge(alpha=0)  # Alpha=0 corresponds to linear regression without penalty
erreur_modele_RLM_complet = -cross_val_score(modele_RLM_complet, StandardScaler().fit_transform(X),
                                             np.log(data['prix']), scoring='neg_mean_squared_error', cv=10).mean()
print(f"Prediction error of the complete RLM model: {erreur_modele_RLM_complet}")

# Lasso regression with cross-validation for finding the best alpha

# Create LassoCV model with 10-fold cross-validation and specified alpha values
lasso_cv = LassoCV(cv=10, alphas=np.logspace(-6, 6, 13))
lasso_cv.fit(StandardScaler().fit_transform(X), np.log(data['prix']))

# Plot the regularization path
alphas_lasso = np.logspace(-6, 6, 13)
plt.figure(figsize=(8, 6))
plt.semilogx(alphas_lasso, lasso_cv.mse_path_.mean(axis=1))
plt.xlabel('Alpha')
plt.ylabel('Mean Squared Error')
plt.title('Regularization Path (Lasso Regression)')
plt.show()

# Best alpha
best_alpha_lasso = lasso_cv.alpha_
print(f"Best alpha (Lasso): {best_alpha_lasso}")

# Coefficients with the best alpha
lasso_coef = lasso_cv.coef_
selected_variables_lasso = np.where(lasso_coef != 0)[0]
selected_variable_names_lasso = X.columns[selected_variables_lasso]
print("\nSelected variables (Lasso):")
print(selected_variable_names_lasso)

# Prediction error of the optimal lasso model
erreur_modele_lasso_opt = lasso_cv.mse_path_.mean(axis=1).min()
print(f"Prediction error of the optimal lasso model: {erreur_modele_lasso_opt}")

# Prediction error of the complete RLM model
modele_RLM_complet = LinearRegression()
erreur_modele_RLM_complet = -cross_val_score(modele_RLM_complet, StandardScaler().fit_transform(X),
                                             np.log(data['prix']), scoring='neg_mean_squared_error', cv=10).mean()

print(f"Prediction error of the complete RLM model: {erreur_modele_RLM_complet}")

########################################################
############ Random Forest Regression Model ############
########################################################

# Create Random Forest model with 100 estimators and specified random state
random_forest = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model to the data
random_forest.fit(X, np.log(data['prix']))

# Make predictions
predictions_rf = random_forest.predict(X)

# Visualize feature importance
feature_importances = random_forest.feature_importances_

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

# Sort by descending order of importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Select important variables (arbitrary threshold)
threshold = 0.02  # Adjust this threshold as needed
selected_variables_rf = importance_df[importance_df['Importance'] > threshold]['Feature'].values

# Display selected variables
print("\nSelected variables (Random Forest):")
print(selected_variables_rf)

# Plot the importance bar with hue
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', hue='Feature', data=importance_df, palette='viridis', legend=False)
plt.title('Feature Importance (Random Forest)')
plt.show()

# Prediction error of the Random Forest model
erreur_modele_rf = -cross_val_score(random_forest, XX, np.log(data['prix']), scoring='neg_mean_squared_error', cv=10).mean()

print(f"Prediction error of the Random Forest model: {erreur_modele_rf}")

# Calculer le coefficient de détermination R²
r2 = r2_score(np.log(data['prix']), predictions_rf)

# Afficher le résultat
print("Le coefficient de détermination R² est :", r2)

################################################################
# We opt for the Random Forest model since it has the optimal  R²
################################################################

# Residual analysis and diagnostics
residuals = np.log(data['prix']) - predictions_rf

# QQ Plot for normality
sm.qqplot(residuals, line='s')
plt.show()

# Autocorrelation function plot
plot_acf(residuals, lags=20)
plt.show()

# Residuals vs. Predictions scatter plot for individual autocorrelation
plt.scatter(predictions_rf, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predictions')
plt.ylabel('Residuals')
plt.show()

# Residuals vs. Predictions scatter plot for homoscedasticity
plt.scatter(predictions_rf, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predictions')
plt.ylabel('Residuals')
plt.show()

# Normality test (Shapiro-Wilk)
_, p_value = shapiro(residuals)
print("Normality test (Shapiro-Wilk) p-value:", p_value)

# Durbin-Watson test
dw_statistic = durbin_watson(residuals)
print("Durbin-Watson test:", dw_statistic)

# Anderson-Darling test
result = anderson(residuals)
print("Anderson-Darling test statistic:", result.statistic)
print("P-value:", result.critical_values)

# Box-Cox transformation
transformed_prices, lambda_value = boxcox(data['prix'])

# Regression model with transformed dependent variable
random_forest.fit(X, np.log(transformed_prices))

# Predictions
predictions_test = random_forest.predict(X)
residualstest = np.log(data['prix']) - predictions_test

# Anderson-Darling test after Box-Cox transformation
result = anderson(residualstest)
print("Anderson-Darling test statistic:", result.statistic)
print("P-value:", result.critical_values)

# Calculate residuals
residuals = np.log(data['prix']) - predictions_rf

# Calculate standardized residuals
standardized_residuals = residuals / np.std(residuals)

# Identify indices of outliers
outlier_indices = np.where(np.abs(standardized_residuals) > 3)[0]

# Remove outliers
X_no_outliers = np.delete(X, outlier_indices, axis=0)
y_no_outliers = np.delete(np.log(data['prix']), outlier_indices)

# Create a new instance of RandomForestRegressor
random_forest_no_outliers = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model without outliers
random_forest_no_outliers.fit(X_no_outliers, y_no_outliers)

# Predictions without outliers
predictions_rf_no_outliers = random_forest_no_outliers.predict(X_no_outliers)

# Calculate new residuals
residuals_no_outliers = y_no_outliers - predictions_rf_no_outliers

# QQ Plot for normality without outliers
sm.qqplot(residuals_no_outliers, line='s')
plt.show()

# Autocorrelation function plot without outliers
plot_acf(residuals_no_outliers, lags=20)
plt.show()

# Scatter plot of residuals vs. predictions without outliers
plt.scatter(predictions_rf_no_outliers, residuals_no_outliers)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predictions')
plt.ylabel('Residuals')
plt.show()

# Normality test without outliers
_, p_value = shapiro(residuals_no_outliers)
print("Normality test without outliers (Shapiro-Wilk) p-value:", p_value)

# Anderson-Darling test without outliers
result = anderson(residuals_no_outliers)
print("Anderson-Darling test statistic without outliers:", result.statistic)
print("P-value:", result.critical_values)

# Box-Cox transformation without outliers
transformed_prices, lambda_value = boxcox(y_no_outliers)

# Regression model with transformed dependent variable without outliers
random_forest.fit(X_no_outliers, transformed_prices)

# Predictions without outliers
predictions_test = random_forest.predict(X_no_outliers)
residualstest = transformed_prices - predictions_test

# Normality test after Box-Cox transformation without outliers
_, p_value = shapiro(residualstest)
print("Normality test without outliers (Shapiro-Wilk) p-value:", p_value)

# Anderson-Darling test after Box-Cox transformation without outliers
result = anderson(residualstest)
print("Anderson-Darling test statistic without outliers:", result.statistic)
print("P-value:", result.critical_values)

# Final Random Forest model without outliers feature importances
feature_importances = random_forest_no_outliers.feature_importances_

# Display variable importance
for i, importance in enumerate(feature_importances):
    print(f"Variable {i+1}: Importance = {importance}")

# Create a DataFrame for the features without outliers
df_X_no_outliers = pd.DataFrame(X_no_outliers)
