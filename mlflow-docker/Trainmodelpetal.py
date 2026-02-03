import mlflow
import pandas as pd
from pandas import get_dummies
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request


df = pd.read_csv("iris.csv", decimal=",")
df = get_dummies(df, columns=["species"], drop_first=True)

X = df.drop(columns=["sepal_length"])
y = df["sepal_length"]
z = df["petal_length"]


X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(X, y,z, test_size=0.6)

mlflow.set_experiment("Mlflow flowers")
mlflow.sklearn.autolog()

# lr = RandomForestRegressor(n_estimators=100, random_state=42)
lr= LinearRegression()
lr.fit(X_train, y_train)
r2 = lr.score(X_test, y_test)
print(f"📈 R² sur le jeu de test : {r2:.2f}")


print("\n---TEST DE PRÉDICTION ---")

fleur_test = pd.DataFrame({
    "sepal_width": [5.1],

    # 0, 0 = Setosa
    "species_versicolor": [0],
    "species_virginica": [0]
})

fleur_test = fleur_test.reindex(columns=X_train.columns, fill_value=0)

resultat = lr.predict(fleur_test)

print(f"Pour une largeur de sépale de {fleur_test['sepal_width'][0]} et l'espèce donnée :")
print(f"🔮 Longueur de sépale prédite : {resultat[0]:.2f}")


app = Flask(__name__)

model_columns = list(X_train.columns)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', prediction_text="")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        width = float(request.form['sepal_width'])
        species = request.form['species']

        is_versicolor = 1 if species == 'versicolor' else 0
        is_virginica = 1 if species == 'virginica' else 0

        input_data = pd.DataFrame({
            "sepal_width": [width],
            "species_versicolor": [is_versicolor],
            "species_virginica": [is_virginica]
        })

        input_data = input_data.reindex(columns=model_columns, fill_value=0)

        prediction = lr.predict(input_data)
        resultat = round(prediction[0], 2)

        return render_template('index.html', prediction_text=f"🔮 Longueur de sépale prédite : {resultat} cm")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Erreur : {str(e)}")

if __name__ == "__main__":
    print("Code d'entraînement terminé. Lancement du serveur web...")
    app.run(debug=True, host='0.0.0.0', port=5002)