"""
Script: main_data_scraping.py
Description: Main script for scraping data from the Spoticar website.
"""

import os
import pandas as pd
from src.tools.data_scraping.scraping_spoticar_functions import scrape_spoticar_data, update_excel_data


# URL of the first page
base_url = "https://www.spoticar.fr/api/vehicleoffers/paginate/search?page="

# Total number of pages to iterate through
num_pages = 1  # You can adjust this number according to your needs

# Scrape data using the function
new_data = scrape_spoticar_data(base_url, num_pages)

# Get the current script's directory
script_directory = os.path.dirname(os.path.abspath(__file__))

# Navigate up one level to access the parent directory of the parent directory (src)
project_directory = os.path.join(script_directory, 'src')

# Construct the absolute path using the project directory and navigate to the data folder
raw_data_path = os.path.join(project_directory, "data", "raw_data.csv")

# Load existing data
try:
    existing_data = pd.read_csv(raw_data_path)
except FileNotFoundError:
    existing_data = pd.DataFrame()

# Update data
update_excel_data(existing_data, new_data, raw_data_path)
