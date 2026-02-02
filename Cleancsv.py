import pandas as pd

df = pd.read_csv("iris.csv", decimal=",")

df = df.drop(columns=["petal_length", "petal_width"])

df.to_csv("iris_clean.csv", decimal=",", index=False)
