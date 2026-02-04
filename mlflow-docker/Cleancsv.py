import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("http://localhost:5432", echo=True)

df = pd.read_csv("iris.csv", decimal=",")

df = df.drop(columns=["petal_length", "petal_width"])

df.to_csv("iris_clean.csv", decimal=",", index=False)



