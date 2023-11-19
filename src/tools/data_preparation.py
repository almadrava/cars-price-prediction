##Importing librairies

import pandas as pd
import os
from unidecode import unidecode

##Importing data
def load_excel_data():
    # Get the current script's directory
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Navigate up one level to access the parent directory (src)
    project_directory = os.path.dirname(script_directory)

    # Construct the absolute path using the project directory
    file_path = os.path.join(project_directory, 'data/raw_data.xlsx')

    # Load the Excel file using pandas
    df = pd.read_excel(file_path)

    # Return the DataFrame
    return df

# Call the function to load the Excel data
raw_data= load_excel_data()

## Cleaning data process

# Replace spaces with hyphens in the "Modele" column
raw_data['Modèle'] = raw_data['Modèle'].str.replace(' ', '-')
# Convert the "Couleur" column to lowercase
raw_data['Couleur'] = raw_data['Couleur'].str.lower()
#Remove accents from "Marque" column
raw_data['Marque'] = raw_data['Marque'].apply(unidecode)


#Exporting cleaned data
def export_cleaned_data(clean_data):
    # Get the current script's directory
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Navigate up one level to access the parent directory (src)
    project_directory = os.path.dirname(script_directory)

    # Construct the relative path for the clean data file
    clean_data_path = os.path.join(project_directory , 'data/clean_data.xlsx')

    # Export the cleaned DataFrame to a new Excel file
    clean_data.to_excel(clean_data_path, index=False)

    # Display a message indicating successful export
    print(f"Cleaned data exported to: {clean_data_path}")

# Call the function to export cleaned data
export_cleaned_data(raw_data)

print(raw_data.dtypes)