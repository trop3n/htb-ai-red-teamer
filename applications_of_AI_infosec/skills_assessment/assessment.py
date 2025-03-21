import requests
import zipfile
import io
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
import sys
import json

def download():
    url = "https://academy.hackthebox.com/storage/modules/292/skills_assessment_data.zip"
    response = requests.get(url)
    if response.status_code == 200:
        print("Download successful")
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall("skills_assessment_data")
            print("Extraction Successful")
    else:
        print("Failed to download the dataset")

def dataset():
    df = pd.read_json("skills_assessment_data/train.json", orient="records")
    df.info()
    # Drop duplicates
    df = drop_duplicates()
    return df

def clean_text(text):
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ",).strip()
    return text

def preprocessing(df):
    df["text"] = df["text"].apply(lambda x: x.lower())
    df["text"] = df["text"] = df.apply(clean_text)
    return df

def train_model(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.3, random_state=42
    )
    pipeline = Pipeline([
        
    ])