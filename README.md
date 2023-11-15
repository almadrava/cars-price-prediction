# Prédiction du prix de revente d'un véhicule d'occasion 
## Objectif du projet 

Le but de cette application est de fournir aux utilisateurs une estimation du prix de revente des véhicules d'occasion en fonction de leurs caractéristiques spécifiques. L'application utilisera un modèle de prédiction entraîné sur des données régulièrement mises à jour garantissant ainsi que les estimations de prix reflètent les tendances actuelles du marché automobile.

## Comment configurer 
### Prérequis
Assurez-vous d'avoir installé Python sur votre machine. Si ce n'est pas le cas, vous pouvez le télécharger depuis le site officiel de Python.
### Installation des dépendances
Exécutez la commande suivante depuis votre terminal pour installer les dépendances nécessaires à l'aide du fichier requirements.txt :

```markdown
pip install -r requirements.txt
```
## Comment éxecuter 
Pour utiliser cette application, suivez les étapes ci-dessous :

**1.** *Lancement de l'Interface Excel :*
   - Accédez au répertoire `src/interfaces`.
   - Exécutez le script VBA `votre_fichier.xlsm` pour ouvrir l'interface Excel.

**2.** *Entrée des Caractéristiques de la Voiture :*
   - Une fois l'interface Excel ouverte, entrez les caractéristiques spécifiques de la voiture dans les champs appropriés.

**3.** *Prédiction du Prix :*
   - Cliquez sur le bouton "Prédire" dans l'interface. Cela déclenchera automatiquement le modèle en arrière-plan, écrit en Python, qui effectuera la prédiction du prix en fonction des caractéristiques entrées.

**4.** *Affichage du Tableau de Bord Excel :*
   - Un tableau de bord Excel s'affichera, montrant le prix estimé de la voiture ainsi que d'autres visualisations relatives aux données des voitures.

**5.** *Mise à Jour des Données :*
   - En cas de besoin, cliquez sur le bouton "MAJ" dans l'interface Excel. Cela déclenchera la mise à jour des données pour le modèle, assurant ainsi que les prédictions sont basées sur des données actualisées.

## Éléments spécifiques sur le développement de l'application

**1.** *Collecte et Préparation des Données :*
   - Collecte des données à partir du site Spoticar via un processus de scrapping.
   - Nettoyage et préparation des données, incluant le changement du type des variables et la gestion des valeurs manquantes.

**2.** *Réalisation de l'Interface VBA :*
   - Conception et développement d'une interface VBA permettant aux utilisateurs d'entrer les caractéristiques spécifiques d'une voiture.

**3.** *Algorithme de Prédiction :*
   - Mise en place d'un premier algorithme de régression linéaire multiple.
   - Application de modèles de sélection des variables pour déterminer les caractéristiques ayant le plus d'impact sur le prix d'un véhicule.
   - Finalisation du modèle de prédiction, y compris des tests unitaires pour garantir la précision de l'algorithme.

**4.** *Lien VBA-Python :*
   - Mise en place de la connexion entre l'interface VBA et le script Python, permettant ainsi à l'interface d'appeler le modèle de prédiction Python pour estimer le prix d'une voiture.
## Stucture du dépôt 
La structure du dépôt est la suivante :

- docs/                    
  - projet.docs : les présentations .pptx de chaque sprint réalisé ainsi que le rapport final de notre projet  
  - demos/ : répertoire contenant les vidéos de démonstration de chaque partie de notre projet
  
- src/               
  - data/ : répertoire contenant tous les fichiers .csv de notre projet
  - interfaces : on trouve ici les fichiers .xlsm de l'interface utilisateur et du tableau de bord 
  - tools : dossier où on retrouve les codes Python (prépartion des données, modèles de prédiction ...)
- tests/  : tous les tests unitaires réalisés sur nos scripts Python              
- requirements.txt  : fichier spécifiant tous les packages nécessaires à l'execution des scripts Python
- README.md          




## Contributeurs

- Colson Benjamin (Product Owner)
- Saiz Edgar (Scrum Master)
- Friaa Isra (Data Engineer)
- Masseau Luxon (Data Scientist)
- Papelard Charlotte (Front/User Interface)

## Contacts

Si vous avez des questions ou des remarques sur l'application, vous pouvez nous contacter ici :[charlotte.papelard@gmail.com], [benjamincolson@hotmail.fr], [edgarsaiz24@gmail.com], [issra.fria@gmail.com], [luxonmasseau@hotmail.com]

Veuillez contacter la personne concernée à votre questionnement, sinon contacter par défaut le Product Owner ou le Scrum Master.

  ----
