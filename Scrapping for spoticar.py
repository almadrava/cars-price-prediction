#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 00:54:00 2023

@author: charlottepapelard
"""
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import pandas as pd

# URL de la première page
base_url = "https://www.spoticar.fr/voitures-occasion?page="

# Nombre total de pages à parcourir
num_pages = 1500  # Vous pouvez ajuster ce nombre selon vos besoins

# Utilisation d'un gestionnaire de contexte pour le navigateur
with webdriver.Chrome() as driver:
    page_contents = []

    for page_num in range(1, num_pages + 1):
        url = base_url + str(page_num)
        driver.get(url)
        time.sleep(5)

        info_containers = driver.find_elements(By.CLASS_NAME, 'card-content')
        
        for info_container in info_containers:
            info = info_container.text.split('\n')
            page_contents.append({
                "Marque": info[0],
                "Style": info[1],
                "Kilométrage": info[2],
                "Carburant": info[3],
                "Année": info[4],
                "Transmission": info[5],
                "Prix": info[6],
                "Prix par mois": info[7] if len(info) > 7 else ""
            })

    df_voiture = pd.DataFrame(page_contents)
    df_voiture.to_csv("spoticar.csv", index=False)

