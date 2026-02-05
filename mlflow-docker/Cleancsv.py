import pandas as pd
import os
from sqlalchemy import create_engine

# --- CONFIGURATION ---
# On récupère les variables d'environnement ou on utilise les défauts
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_NAME = os.getenv("DB_NAME", "mlflow")
# Dans Docker, l'hôte est le nom du service : "db"
DB_HOST = "db" 

# Correction de l'URL : protocole postgresql et hôte correct
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:5432/{DB_NAME}"
engine = create_engine(DATABASE_URL, echo=True)

# --- TRAITEMENT ---
try:
    # Lecture du fichier original
    df = pd.read_csv("iris.csv", decimal=",")

    # Nettoyage des colonnes
    df = df.drop(columns=["petal_length", "petal_width"])

    # Sauvegarde en fichier CSV local pour le service Trainmodel
    df.to_csv("iris_clean.csv", decimal=",", index=False)
    print("✅ Fichier iris_clean.csv généré avec succès.")

    # Optionnel : Sauvegarde aussi dans la base de données
    df.to_sql('iris_raw_data', engine, if_exists='replace', index=False)
    print("✅ Données brutes injectées dans PostgreSQL.")

except Exception as e:
    print(f"❌ Erreur lors du nettoyage : {e}")