import pandas as pd
import mlflow
import os # <--- On importe 'os' pour lire les variables système
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("Chargement des données...")
# Assurez-vous que iris.csv est bien dans le dossier ou copié par le Dockerfile
df = pd.read_csv("iris.csv") 

print("Encodage de la colonne species...")
df_encoded = pd.get_dummies(df, columns=["species"])

print("Séparation X/y...")
X = df_encoded.drop("sepal_length", axis=1)
y = df_encoded["sepal_length"]

print("Split train/test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Initialisation MLflow...")

# --- MODIFICATION ICI ---
# On récupère l'adresse définie dans docker-compose.yml
# Si elle n'existe pas, on prend "http://mlflow:5000" par défaut
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("iris_experiment")
# ------------------------

print(f"Tracking URI utilisé : {tracking_uri}")

print("Entraînement du modèle...")
with mlflow.start_run():
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Prédiction...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse}, R2: {r2}")
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)
    mlflow.sklearn.log_model(model, "model")
    print("Fin du run MLflow.")