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

