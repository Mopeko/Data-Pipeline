import os
import mlflow
import pandas as pd
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Float, String, Integer, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError

# --- 1. CONFIGURATION BASE DE DONNÉES ---
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_NAME = os.getenv("DB_NAME", "mlflow")
DB_HOST = "db"

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:5432/{DB_NAME}"

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- CHANGEMENT 1 : Nouvelle classe pour une NOUVELLE table ---
class PredictionNoSpecies(Base):
    # On donne un nom différent pour ne pas toucher à l'ancienne table 'predictions'
    __tablename__ = "predictions_no_species"
    
    id = Column(Integer, primary_key=True, index=True)
    sepal_width = Column(Float)
    predicted_length = Column(Float)
    # Plus de colonne species ici !

def init_db():
    retries = 5
    while retries > 0:
        try:
            # SQLAlchemy va voir qu'il y a une nouvelle table à créer
            # Il ne touchera pas aux anciennes
            Base.metadata.create_all(bind=engine)
            print("✅ Connexion réussie & Tables mises à jour !")
            return True
        except OperationalError:
            retries -= 1
            print(f"Waiting for database... ({retries} retries left)")
            time.sleep(3)
    return False

init_db()

# --- 2. LOGIQUE D'ENTRAÎNEMENT ---

def prepare_model_and_data():
    try:
        df = pd.read_csv("iris_clean.csv", decimal=",")
    except:
        df = pd.read_csv("iris_clean.csv")

    # Suppression de la colonne species pour le traitement
    if "species" in df.columns:
        df = df.drop(columns=["species"])

    # --- CHANGEMENT 2 : Stockage dans une NOUVELLE table ---
    try:
        inspector = inspect(engine)
        # On vérifie si la table V2 existe, sinon on la crée
        if not inspector.has_table('iris_data_no_species'):
            df.to_sql('iris_data_no_species', engine, if_exists='replace', index=False)
            print("✅ Dataset (sans espèces) stocké dans 'iris_data_no_species'.")
        else:
            print("ℹ️ La table 'iris_data_no_species' existe déjà.")

    except Exception as e:
        print(f"⚠️ Erreur stockage dataset: {e}")

    # Définition X et y
    X = df.drop(columns=["sepal_length"])
    y = df["sepal_length"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("Mlflow No Species")
    mlflow.sklearn.autolog()

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr, list(X_train.columns)

model, model_columns = prepare_model_and_data()

# --- 3. CONFIGURATION FASTAPI ---
app = FastAPI(title="Iris Predictor V2")
templates = Jinja2Templates(directory="templates")

@app.get("/styles.css")
async def styles():
    return FileResponse("templates/styles.css")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction_text": ""})

@app.post("/predict_alt", response_class=HTMLResponse)
async def predict(request: Request, sepal_width: float = Form(...)):
    db = SessionLocal()
    try:
        # Préparation des données
        input_data = pd.DataFrame({"sepal_width": [sepal_width]})
        input_data = input_data.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(input_data)
        resultat = float(round(prediction[0], 2))

        # --- CHANGEMENT 3 : Enregistrement dans la NOUVELLE table ---
        new_pred = PredictionNoSpecies(
            sepal_width=sepal_width, 
            predicted_length=resultat
        )
        db.add(new_pred)
        db.commit()

        return templates.TemplateResponse("index.html", {"request": request, "prediction_text": str(resultat)})

    except Exception as e:
        db.rollback()
        print(f"Erreur Predict: {e}")
        return templates.TemplateResponse("index.html", {"request": request, "prediction_text": f"Erreur : {str(e)}"})
    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5002)