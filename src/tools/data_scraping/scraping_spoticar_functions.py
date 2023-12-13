#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 13:29:53 2023

@author: charlottepapelard
"""
"""
Module: data_scraping_functions.py
Description: Contains functions for scraping vehicle data from the Spoticar website.
"""
# Importing necessary librairies
from lxml import etree
import pandas as pd
from selenium import webdriver
import json
from datetime import datetime
import os


def clean(stri):
    """
    Cleans a string by removing specified characters.

    Parameters:
    - stri (str): The input string to be cleaned.

    Returns:
    str: The cleaned string with specified characters removed.
    """
    stri = stri.replace("['", "").replace("']", "").replace("[", "").replace("]", "")
    return stri

def scrape_spoticar_data(base_url, num_pages):
    """
    Scrapes vehicle data from the Spoticar website.

    Parameters:
    - base_url (str): The base URL for the API endpoint.
    - num_pages (int): Total number of pages to iterate through.

    Returns:
    pd.DataFrame: A DataFrame containing scraped vehicle details.
    """
    # Using a context manager for the browser
    with webdriver.Chrome() as driver:
        page_contents = []

        for page_num in range(1, num_pages + 1):
            url = base_url + str(page_num)
            driver.get(url)

            # Retrieve the page source using execute_script
            page_source = driver.execute_script("return document.body.innerHTML")
            dom = etree.HTML(page_source)

            # Convert the page source to a dictionary (JSON)
            json_data = json.loads(dom.xpath("//pre")[0].text)
            cars = json_data['hits']

            for car in cars:
                details = car.get('_source', {})
                page_contents.append({
                    "marque": details.get('marque', [None])[0],
                    "modele": details.get('model', [None])[0],
                    "marque_et_modele": details.get('ligne', [None])[0],
                    "boite_vitesse": details.get('boite_vitesse', [None])[0],
                    "couleur": details.get('color', [None])[0],
                    "critair": details.get('field_green_zone_level', [None])[0],
                    "categorie": details.get('field_vo_categories', [None])[0],
                    "cylindree": details.get('field_vo_cylindree', [None])[0],
                    "kilometrage": details.get('field_vo_km', [None])[0],
                    "nb_places": details.get("field_vo_nb_places", [None])[0],
                    "nb_portes": details.get("field_vo_nb_portes", [None])[0],
                    "nb_vitesses": details.get("field_vo_nb_vitesses", [None])[0],
                    "puissance_fiscale": details.get("field_vo_puissance_fiscale", [None])[0],
                    "puissance_physique": details.get("field_vo_puissance_physique", [None])[0],
                    "carburant": details.get('type_carburant', [None])[0],
                    "annee": details.get('field_vo_dpi', [None])[0],
                    "nb_roues_motrices": details.get('transmission', [None])[0],
                    "prix": details.get('field_vo_prix_base', [None])[0],
                })

    return pd.DataFrame(page_contents)


def update_excel_data(existing_data, new_data, new_directory):
    """
    Update existing CSV data with new data and store the result.

    Parameters:
    - existing_data (pd.DataFrame): Existing DataFrame to be updated.
    - new_data (pd.DataFrame): New DataFrame to be added.
    - archive_directory (str): Directory where archived raw_data files are stored.
    """

    # Concatenate existing data with new data
    updated_data = pd.concat([existing_data, new_data], ignore_index=True, axis=0).reset_index(drop=True)

    # Get the current date and time
    current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")

    # Construct the file name with the current date and time
    raw_data_path = f"raw_data_{current_datetime}.csv"

    # Store the updated data in a new CSV file
    updated_data.to_csv(os.path.join(new_directory, raw_data_path), index=False)
