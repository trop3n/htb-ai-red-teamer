import pandas as pd

# load the dataset
data = pd.read_csv("./demo_dataset.csv")
print(data.head())
print(data.info())
print(data.isnull().sum())