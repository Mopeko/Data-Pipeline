import os
import mlflow
import pandas as pd
import time
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

# -----------------------------
# 1️⃣ CONFIGURATION BASE DE DONNÉES
# -----------------------------
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_NAME = os.getenv("DB_NAME", "mlflow")
DB_HOST = "db"

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:5432/{DB_NAME}"

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Modèle A → table avec species
class PredictionEntry(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    species = Column(String)
    sepal_width = Column(Float)
    predicted_length = Column(Float)

# Modèle B → table sans species
class PredictionNoSpecies(Base):
    __tablename__ = "predictions_no_species"
    id = Column(Integer, primary_key=True, index=True)
    sepal_width = Column(Float)
    predicted_length = Column(Float)

# Init DB
def init_db():
    retries = 5
    while retries > 0:
        try:
            Base.metadata.create_all(bind=engine)
            print("✅ Connexion réussie & Tables mises à jour !")
            return True
        except OperationalError:
            retries -= 1
            print(f"Waiting for database... ({retries} retries left)")
            time.sleep(3)
    return False

init_db()

# -----------------------------
# 2️⃣ LOGIQUE D'ENTRAÎNEMENT DES DEUX MODÈLES
# -----------------------------
def prepare_model_A():
    df = pd.read_csv("iris_clean.csv", decimal=",")
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

def prepare_model_B():
    df = pd.read_csv("iris_clean.csv", decimal=",")
    X = df[["sepal_width"]]  # uniquement sepal_width
    y = df["sepal_length"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("Mlflow no species")
    mlflow.sklearn.autolog()

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr, list(X_train.columns)

model_A, columns_A = prepare_model_A()
model_B, columns_B = prepare_model_B()

# -----------------------------
# 3️⃣ FASTAPI
# -----------------------------
app = FastAPI(title="Iris Predictor Dual Model")
templates = Jinja2Templates(directory="templates")

@app.get("/styles.css")
async def styles():
    return FileResponse("templates/styles.css")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction_text": ""})

# Route modèle A
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
        }).reindex(columns=columns_A, fill_value=0)

        prediction = model_A.predict(input_data)
        resultat = float(round(prediction[0], 2))

        new_pred = PredictionEntry(species=species, sepal_width=sepal_width, predicted_length=resultat)
        db.add(new_pred)
        db.commit()

        return templates.TemplateResponse("index.html", {"request": request, "prediction_text": str(resultat)})
    finally:
        db.close()

# Route modèle B
@app.post("/predict_alt", response_class=HTMLResponse)
async def predict_alt(request: Request, sepal_width: float = Form(...)):
    db = SessionLocal()
    try:
        input_data = pd.DataFrame({"sepal_width": [sepal_width]}).reindex(columns=columns_B, fill_value=0)
        prediction = model_B.predict(input_data)
        resultat = float(round(prediction[0], 2))

        new_pred = PredictionNoSpecies(sepal_width=sepal_width, predicted_length=resultat)
        db.add(new_pred)
        db.commit()

        return templates.TemplateResponse("index.html", {"request": request, "prediction_text": str(resultat)})
    finally:
        db.close()

# -----------------------------
# 4️⃣ LANCEMENT UVICORN
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5002, reload=True)
