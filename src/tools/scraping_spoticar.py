#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 13:29:53 2023

@author: charlottepapelard
"""

from lxml import etree
import pandas as pd
from selenium import webdriver
import json 

def clean(stri):
    stri = stri.replace("['","").replace("']","").replace("[","").replace("]","")
    return stri

# URL of the first page
base_url = "https://www.spoticar.fr/api/vehicleoffers/paginate/search?page="

# Total number of pages to iterate through
num_pages = 1 # You can adjust this number according to your needs

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
        
raw_data = pd.DataFrame(page_contents)

# Store the data in an Excel file
raw_data.to_excel("raw_data.xlsx", index=False)
