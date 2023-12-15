## Importing librairies

from unidecode import unidecode
import os
import pandas as pd

## Importing the data


def load_excel_data():
    """
    Load the latest  data from the CSV file and return a DataFrame.

    Returns:
    pandas.DataFrame: The DataFrame containing the latest data loaded from the CSV file.
    """
    # Get the current script's directory
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Navigate up two levels to access the parent directory of the parent directory (src)
    project_directory = os.path.dirname(os.path.dirname(script_directory))

    # Construct the absolute path using the project directory and navigate to the archived_data folder
    archived_data_directory = os.path.join(project_directory, "data", "latest_data")

    # Get the list of files in the latest_data directory
    files = os.listdir(archived_data_directory)

    # Filter out non-CSV files
    csv_files = [file for file in files if file.endswith(".csv")]

    # Sort the CSV files by modification time to get the latest one
    latest_csv_file = max(csv_files, key=lambda x: os.path.getmtime(os.path.join(archived_data_directory, x)))

    # Construct the absolute path to the latest CSV file
    latest_csv_path = os.path.join(archived_data_directory, latest_csv_file)

    # Load the latest CSV file using pandas
    df = pd.read_csv(latest_csv_path)

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
    clean_data_path = os.path.join(project_directory , 'data/clean_data.csv')

    # Export the cleaned DataFrame to a new Excel file
    clean_data.to_csv(clean_data_path, index=False)



# Call the function to export cleaned data
#export_cleaned_data(raw_data)