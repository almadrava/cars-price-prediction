# Prédiction du prix d'un véhicule en fonction de ses caractéristiques (Projet académique)

Malgré le fait que de plus en plus de personnes changent leurs habitudes au niveau des transports en passant au vélo ou transports en commun. La voiture, quelle soit électrique ou non, reste le moyen de transport le plus utilisé dans le monde et d'autant plus en France. Il n'y a pas moins de 37 880 000 de voitures particulières présentes sur le territoire français. De plus, la centralisation des institutions dans les grandes villes poussent les jeunes provenant des zones rurales à se munir de voitures afin d'effectuer les trajets de la maison familiale à leur lieu d'études ou travail. Cependant de nombreux étudiants réfléchissent à changer de voiture une fois rentré dans le monde professionnel, ils vendent ainsi leur véhicule actuel afin de'avoir les fonds suffisants pour leur futurs achats. Grâce à notre application, les étudiants ou tout autre personne désirant se séparer de leur bien automobiles pourront avoir une estimation de la somme qu'ils peuvent espérer obtenir suite à la vente.

Ce projet est effectué dans le cadre de la matière "Gestion de Projet et Produit Digital" lors de la 2ème année du Master SEP (Statistique pour l'Evaluation et la Prévison) 2023/2024 à l'Université de Reims Champagne-Ardenne.

## Structure et méthode choisie

Ce projet se réalise à l'aide de deux logiciels de programmation qui sont VBA et Python, de nombreux liens seront fait entre les deux logiciels:

1) Créaton de la base de données Excel : A l'aide d'une execution d'un script python lancé depuis Excel lançant un scrapping afin d'obtenir les données nécessaires à la  prédiction.
2) Création d'un Formulaire en VBA : Cette deuxième étape consiste à développer un formulaire en VBA qui servira d'interface utilisateur. Ce formulaire permettra aux utilisateurs de saisir des données relatives à un vehicule, telles que sa marque, son modèle, les caractéristiques lié au modèle (nombre de chevaux, options, etc...) et le parcours de vie du vehicule (kilométrage, première main, etc...). Les données saisies par les utilisateurs seront collectées, formatées et stockées.
3) Lancement de la prédiction du prix : Suite aux données récupérées par le formulaire, le script de prédiction sera lancé récupérant les variables à utilisées par sélection de modèles, régressions logistiques et autres procédés statistiques. Une fois nos variables explicatives récupérées, elles seront utilisés dans la prédiction du prix du véhicule de l'utilisateur.
4) Création du Dashboard : Python renverra à Excel le prix obtenu et ainsi un dashboard adapté sera proposé à l'utilisateur afin de lui expliquer le prix et de lui permettre de faire une comparaison simple de son véhicule avec les autres.

A noter que l'utilisateur peut faire une prédiction autant de fois qu'il le souhaite car elles seront enrigistré dans la base. Cepandant ces premières prédictions seront perdues une fois la base mise à jour par le nouveau scrapping.



Prédiction du Prix du Vehicule : Les données saisies par les utilisateurs enrichiront notre base de données d'origine, une base de données de référence regroupant un nombre élevé de vehicules. Ces données seront utilisées comme entrées pour des algorithmes Python. Ces algorithmes appliqueront des modèles de prédiction du prix de revente possible etc..

Création d'un Tableau de Bord : . Les résultats de la prédiction seront restitués aux utilisateurs sous forme d'un tableau de bord interactif contenant des visualisations des données, notamment des informations sur le prix du vehicule vis-à-vis des vehicules de même marque etc.

## Installation

## Utilisation

Pour tous savoir sur l'utilisation de l'application, nous vous invitons à lire la notice suivante : (à rajouter lien )

## Contributeurs

- Colson Benjamin (Product Owner)
- Saiz Edgar (Scrum Master)
- Friaa Isra (Data Engineer)
- Masseau Luxon (Data Scientist)
- Papelard Charlotte (Front/User Interface)

## Contacts

Si vous avez des questions ou des remarques sur l'application, vous pouvez nous contacter ici :[benjamincolson@hotmail.fr], [edgarsaiz24@gmail.com], [issra.fria@gmail.com], [luxonmasseau@hotmail.com],[]

Veillez contacter la personne concernée à votre questionnement, sinon contacter par défault le Product Owner ou le Scrum Master.

  ----
