import os
import mlflow
import pandas as pd
from pandas import get_dummies
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Imports FastAPI
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, FileResponse # Ajout de FileResponse
from fastapi.templating import Jinja2Templates

# --- 1. ENTRAÎNEMENT DU MODÈLE ---

# Chargement et préparation des données
df = pd.read_csv("iris_clean.csv", decimal=",")
df = get_dummies(df, columns=["species"], drop_first=True)

X = df.drop(columns=["sepal_length"])
y = df["sepal_length"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

# Configuration MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
mlflow.set_experiment("Mlflow flowers")
mlflow.sklearn.autolog()

# Entraînement
lr = LinearRegression()
lr.fit(X_train, y_train)
model_columns = list(X_train.columns)

# --- 2. CONFIGURATION FASTAPI ---

app = FastAPI(title="Iris Predictor Prototype")

# On définit le dossier des templates
templates = Jinja2Templates(directory="templates")

# NOUVELLE ROUTE : Pour servir le CSS depuis le dossier templates
@app.get("/styles.css")
async def styles():
    return FileResponse("templates/styles.css")

# --- 3. ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Affiche la page d'accueil"""
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "prediction_text": ""}
    )

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request, 
    sepal_width: float = Form(...), 
    species: str = Form(...)
):
    """Gère la soumission du formulaire et renvoie la prédiction"""
    try:
        is_versicolor = 1 if species == 'versicolor' else 0
        is_virginica = 1 if species == 'virginica' else 0

        input_data = pd.DataFrame({
            "sepal_width": [sepal_width],
            "species_versicolor": [is_versicolor],
            "species_virginica": [is_virginica]
        })

        input_data = input_data.reindex(columns=model_columns, fill_value=0)

        prediction = lr.predict(input_data)
        resultat = round(prediction[0], 2)

        return templates.TemplateResponse(
            "index.html", 
            {"request": request, "prediction_text": str(resultat)}
        )

    except Exception as e:
        return templates.TemplateResponse(
            "index.html", 
            {"request": request, "prediction_text": f"Erreur : {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5002)