#!/usr/bin/env python
# coding: utf-8

# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
data1 =  pd.read_excel("D:/clean_data.xlsx")
# Affichage de la matrice de corrélation avec Seaborn


# In[4]:


X_quanti = data1.select_dtypes(include=['int64', 'float64'])


# In[5]:


correlation_matrix = X_quanti.corr()


# In[6]:


correlation_matrix


# In[7]:


sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Matrice de Corrélation')
plt.show()


# In[8]:


X= data1.drop('prix',axis=1).drop('marque',axis=1).drop('modele',axis=1).drop('marque_et_modele',axis=1).drop('couleur',axis=1).drop('nb_roues_motrices',axis=1)


# In[18]:


#Conversion variable qualitative en  booléen
data = pd.get_dummies(data1, columns=['boite_vitesse', 'categorie','critair','carburant'])

# description de la base
print("\ndata.dtypes")
print(data.dtypes)
print("\ndata.head()")
print(data.head())

#Régression linéaire multiple (RLM)
X_numeric = data[['cylindree', 'kilometrage', 'nb_places', 'nb_portes', 'nb_vitesses',
                  'puissance_fiscale', 'puissance_physique', 'annee']]
X_bool = data[['boite_vitesse_manuelle', 'categorie_break', 'categorie_citadine', 'categorie_coupe-cabriolet', 'categorie_familiale', 'categorie_monospace', 'categorie_suv-4x4', 'categorie_utilitaire','critair_niveau 0'
              ,'critair_niveau 1','critair_niveau 2','carburant_diesel','carburant_electrique','carburant_essence','carburant_hybride']].astype(int)
X_prix = data[['prix']]

X = pd.concat([X_numeric, X_bool], axis=1)


# In[19]:


X


# In[20]:


correlation_matrix = X.corr()


# In[21]:


correlation_matrix


# In[22]:


sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Matrice de Corrélation')
plt.show()


# In[23]:


data1.columns


# In[24]:


high_corr_matrix = correlation_matrix[correlation_matrix.abs() > 0.7]

# Affichage de la matrice de corrélation avec Seaborn
sns.heatmap(high_corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Matrice de Corrélation (coefficients > 0.7)')
plt.show()


# In[ ]:




