## Importing librairies

import pandas as pd
import os
from unidecode import unidecode

## Importing the data
def load_excel_data():
    """
        Load data from the Excel file and return a DataFrame.

        Returns:
        pandas.DataFrame: The DataFrame containing the data loaded from the Excel file.
    """
    # Get the current script's directory
    script_directory = os.path.dirname(os.path.abspath(__file__))
    #print(script_directory)

    # Navigate up two levels to access the parent directory of the parent directory (src)
    project_directory = os.path.dirname(os.path.dirname(script_directory))
    #print(project_directory)

    # Construct the absolute path using the project directory and navigate to the data folder
    raw_data_path = os.path.join(project_directory, "data", "raw_data.xlsx")
    #print(raw_data_path)

    # Load the Excel file using pandas
    df = pd.read_excel(raw_data_path)

    # Return the DataFrame
    return df

# Call the function to load the Excel data
#raw_data= load_excel_data()

##Cleaning the data
def clean_data(raw_data):
    """
    Clean the data by performing several cleaning steps.

    Args:
    raw_data (pandas.DataFrame): The DataFrame containing raw data.

    Returns:
    pandas.DataFrame: The DataFrame containing cleaned data.
    """
    # Remove duplicates
    raw_data.drop_duplicates(inplace=True)

    # Drop rows with missing values
    raw_data.dropna(inplace=True)

    # Replace spaces with hyphens
    raw_data['modele'] = raw_data['modele'].str.replace(' ', '-')
    raw_data['marque_et_modele'] = raw_data['marque_et_modele'].str.replace(' ', '-')

    # Convert the "couleur" column to lowercase
    raw_data['couleur'] = raw_data['couleur'].str.lower()

    # Remove accents
    raw_data['marque'] = raw_data['marque'].apply(unidecode)
    raw_data['marque_et_modele'] = raw_data['marque_et_modele'].apply(unidecode)
    raw_data['categorie'] = raw_data['categorie'].apply(unidecode)

    return raw_data
# Apply the function
#raw_data=clean_data(raw_data)
def extract_and_convert(row):
    """
        Extracts and converts a number from a string.

        Args:
        row (str): The string from which to extract the number.

        Returns:
        int or None: The extracted number or None if the conversion fails.
    """
    try:
        # Use a regular expression to extract the number
        number = int(''.join(filter(str.isdigit, row)))
        return number
    except ValueError:
        # Handle cases where the conversion to integer fails
        return None

# Apply the function to the column "nb_roues_motrices"
#raw_data['nb_roues_motrices'] = raw_data['nb_roues_motrices'].apply(extract_and_convert)
# Apply the function to the column "critair"
#raw_data['critair'] = raw_data['critair'].apply(extract_and_convert)


## Exporting cleaned data
def export_cleaned_data(clean_data):
    """
        Exports the cleaned data to an Excel file.

        Args:
        clean_data (pandas.DataFrame): The DataFrame containing the cleaned data.
    """

    # Get the current script's directory
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Navigate up two levels to access the parent directory (src)
    project_directory = os.path.dirname(os.path.dirname(script_directory))

    # Construct the relative path for the clean data file
    clean_data_path = os.path.join(project_directory , 'data/clean_data.xlsx')

    # Export the cleaned DataFrame to a new Excel file
    clean_data.to_excel(clean_data_path, index=False)



# Call the function to export cleaned data
#export_cleaned_data(raw_data)