import mlflow
import pandas as pd
from pandas import get_dummies
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("iris_clean.csv", decimal=",")
df = get_dummies(df, columns=["species"], drop_first=True)

X = df.drop(columns=["sepal_length"])
y = df["sepal_length"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6,random_state=42)

# mlflow.set_tracking_uri("http://localhost:5000")
# Enable autologging for scikit-learn

mlflow.set_experiment("Mlflow flowers")
mlflow.sklearn.autolog()

# Just train the model normally
lr = LinearRegression()
lr.fit(X_train, y_train)


print("\n--- 🧪 TEST DE PRÉDICTION ---")

# 1. On définit la fleur à tester.
# Le modèle a été entraîné UNIQUEMENT sur sepal_width et l'espèce.
fleur_test = pd.DataFrame({
    "sepal_width": [3.5],       # La seule feature numérique disponible dans ton CSV
    
    # Gestion des espèces (drop_first=True a supprimé 'setosa' qui est la référence)
    # 0, 0 -> Setosa
    # 1, 0 -> Versicolor
    # 0, 1 -> Virginica
    "species_versicolor": [0],  
    "species_virginica": [0]
})

# 2. Sécurité : On force l'ordre des colonnes
fleur_test = fleur_test.reindex(columns=X_train.columns, fill_value=0)

# 3. Calcul de la prédiction
resultat = lr.predict(fleur_test)

print(f"Pour une largeur de sépale de {fleur_test['sepal_width'][0]} et l'espèce donnée :")
print(f"🔮 Longueur de sépale prédite : {resultat[0]:.2f}")


# ==========================================
# À RAJOUTER À LA FIN DE TON FICHIER
# ==========================================

from flask import Flask, render_template, request

app = Flask(__name__)

# On sauvegarde la liste des colonnes utilisées lors de l'entraînement
# pour s'assurer que l'input du site web aura exactement le même ordre.
model_columns = list(X_train.columns)

@app.route('/', methods=['GET'])
def index():
    # Affiche la page HTML (doit être dans le dossier templates/index.html)
    return render_template('index.html', prediction_text="")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Récupération des données du formulaire HTML
        # Attention : le name dans le HTML doit correspondre (sepal_width, species)
        width = float(request.form['sepal_width'])
        species = request.form['species']

        # 2. Encodage manuel (car le modèle attend des colonnes spécifiques)
        # Rappel : drop_first=True a été utilisé, donc :
        # Setosa = 0, 0
        # Versicolor = 1, 0
        # Virginica = 0, 1
        
        is_versicolor = 1 if species == 'versicolor' else 0
        is_virginica = 1 if species == 'virginica' else 0

        # 3. Création du DataFrame pour la prédiction
        input_data = pd.DataFrame({
            "sepal_width": [width],
            "species_versicolor": [is_versicolor],
            "species_virginica": [is_virginica]
        })

        # 4. Sécurité : On force l'ordre des colonnes pour qu'il corresponde à X_train
        # fill_value=0 permet de boucher les trous si jamais une colonne manque
        input_data = input_data.reindex(columns=model_columns, fill_value=0)

        # 5. Prédiction
        prediction = lr.predict(input_data)
        resultat = round(prediction[0], 2)

        return render_template('index.html', prediction_text=f"🔮 Longueur de sépale prédite : {resultat} cm")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Erreur : {str(e)}")

# Lancement du serveur
# On utilise host='0.0.0.0' pour que Docker laisse passer la connexion
if __name__ == "__main__":
    print("Code d'entraînement terminé. Lancement du serveur web...")
    app.run(debug=True, host='0.0.0.0', port=5002)