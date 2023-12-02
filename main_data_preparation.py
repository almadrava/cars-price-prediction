# main_data_prep.py

from src.tools.data_preparation.data_preparation_functions import load_excel_data, clean_data, extract_and_convert, export_cleaned_data

def main():
    """
    Main script for data preparation.

    This script loads raw data from an Excel file, performs cleaning steps,
    extracts and converts specific columns, and exports the cleaned data to a new Excel file.

    Usage:
    Run this script to execute the data preparation process.

    Returns:
    None
    """
    # Call the function to load the Excel data
    raw_data = load_excel_data()

    # Apply the function to clean the data
    raw_data = clean_data(raw_data)

    # Apply the function to extract and convert data in specific columns if needed
    raw_data['nb_roues_motrices'] = raw_data['nb_roues_motrices'].apply(extract_and_convert)
    raw_data['critair'] = raw_data['critair'].apply(extract_and_convert)

    # Call the function to export cleaned data
    export_cleaned_data(raw_data)

# Check if the script is executed directly
if __name__ == "__main__":
    main()
