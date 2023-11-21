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

# Remove duplicates in the original dataframe
raw_data.drop_duplicates(inplace=True)

# Drop rows with missing values
raw_data.dropna(inplace=True)

# Replace spaces with hyphens in the "modele" column
raw_data['modele'] = raw_data['modele'].str.replace(' ', '-')
# Replace spaces with hyphens in the "marque_et_modele" column
raw_data['marque_et_modele'] = raw_data['marque_et_modele'].str.replace(' ', '-')
# Convert the "couleur" column to lowercase
raw_data['couleur'] = raw_data['couleur'].str.lower()
#Remove accents from "marque" column
raw_data['marque'] = raw_data['marque'].apply(unidecode)
#Remove accents from "marque_et_modele" column
raw_data['marque_et_modele'] = raw_data['marque_et_modele'].apply(unidecode)
#Remove accents from "categorie" column
raw_data['categorie'] = raw_data['categorie'].apply(unidecode)

# Function to extract only the number from the column and convert it to an integer
def extract_and_convert(row):
    try:
        # Use a regular expression to extract the number
        number = int(''.join(filter(str.isdigit, row)))
        return number
    except ValueError:
        # Handle cases where the conversion to integer fails
        return None

# Apply the function to the column "nb_roues_motrices"
raw_data['nb_roues_motrices'] = raw_data['nb_roues_motrices'].apply(extract_and_convert)
# Apply the function to the column "critair"
raw_data['critair'] = raw_data['critair'].apply(extract_and_convert)


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
    #print(f"Cleaned data exported to: {clean_data_path}")

# Call the function to export cleaned data
export_cleaned_data(raw_data)
