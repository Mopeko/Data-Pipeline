import os
import mlflow
import pandas as pd
import time # Pour la pause en cas d'échec de connexion
from pandas import get_dummies
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
# On sécurise les variables pour éviter le crash "None"
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_NAME = os.getenv("DB_NAME", "mlflow")
DB_HOST = "db" # Dans Docker, on utilise le nom du service

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:5432/{DB_NAME}"

# Tentative de connexion avec Retry (car Postgres met du temps à démarrer)
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PredictionEntry(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    species = Column(String)
    sepal_width = Column(Float)
    predicted_length = Column(Float)

# On attend que la DB soit prête avant de créer les tables
def init_db():
    retries = 5
    while retries > 0:
        try:
            Base.metadata.create_all(bind=engine)
            print("✅ Connexion à la base de données réussie !")
            return True
        except OperationalError:
            retries -= 1
            print(f"Waiting for database... ({retries} retries left)")
            time.sleep(3)
    return False

init_db()

# --- 2. LOGIQUE D'ENTRAÎNEMENT ---

def prepare_model_and_data():
    df = pd.read_csv("iris_clean.csv", decimal=",")
    
    # Stockage initial sécurisé
    try:
        inspector = inspect(engine)
        if not inspector.has_table('iris_cleaned_data'):
            df.to_sql('iris_cleaned_data', engine, if_exists='replace', index=False)
            print("✅ Dataset stocké en base.")
    except Exception as e:
        print(f"⚠️ Erreur stockage dataset: {e}")

    df_dummies = get_dummies(df, columns=["species"], drop_first=True)
    X = df_dummies.drop(columns=["sepal_length"])
    y = df_dummies["sepal_length"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("Mlflow flowers")
    mlflow.sklearn.autolog()

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr, list(X_train.columns)

model, model_columns = prepare_model_and_data()

# --- 3. CONFIGURATION FASTAPI ---
app = FastAPI(title="Iris Predictor Pro")
templates = Jinja2Templates(directory="templates")

@app.get("/styles.css")
async def styles():
    return FileResponse("templates/styles.css")

# --- 4. ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction_text": ""})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, sepal_width: float = Form(...), species: str = Form(...)):
    db = SessionLocal()
    try:
        is_versicolor = 1 if species == 'versicolor' else 0
        is_virginica = 1 if species == 'virginica' else 0

        input_data = pd.DataFrame({
            "sepal_width": [sepal_width],
            "species_versicolor": [is_versicolor],
            "species_virginica": [is_virginica]
        }).reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(input_data)
        resultat = float(round(prediction[0], 2))

        new_pred = PredictionEntry(species=species, sepal_width=sepal_width, predicted_length=resultat)
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