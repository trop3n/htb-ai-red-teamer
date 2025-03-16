import requests
import zipfile
import io
import os
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

# URL of the dataset
# url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
# # Download the dataset
# response = requests.get(url)
# if response.status_code == 200:
#     print("Download successful")
# else:
#     print("Failed to download the dataset")

# # extract the dataset
# with zipfile.ZipFile(io.BytesIO(response.content)) as z:
#     z.extractall("sms_spam_collection")
#     print("Extraction successful")

# list extracted files
# extracted_files = os.listdir("sms_spam_collection")
# print("Extracted files:", extracted_files)

# load the dataset
df = pd.read_csv(
    "sms_spam_collection/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
)

# display basic information about the dataset
print("-------------------- HEAD --------------------")
print(df.head())
print("-------------------- DESCRIBE --------------------")
print(df.describe())
print("-------------------- INFO --------------------")
print(df.info())

# check for missing values
print("Missing values:\n", df.isnull().sum())

# check for duplicates
print("Duplicate entries:", df.duplicated().sum())

# remove any duplicates if any
df = df.drop_duplicates()

# Download necessary NLTK data files
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

print("=== BEFORE ANY PREPROCESSING ===")
print(df.head(5))

# convert all message text to lowercase
df["message"] = df["message"].str.lower()
print("\n=== AFTER LOWERCASING ===")
print(df["message"].head(5))

# Remove non-essential punctuation and numbers, keep useful symbols like $ and !
df["message"] = df["message"].apply(lambda x: re.sub(r"[^a-z\s$!]", "", x))
print("\n=== AFTER REMOVING PUNCTUATION & NUMBERS (except $ and !) ===")
print(df["message"].head(5))

# Split each message into individual tokens
df["message"] = df["message"].apply(word_tokenize)
print("\n=== AFTER TOKENIZATION ===")
print(df["message"].head(5))

# define a set of English stop words and remove them from the tokens
stop_words = set(stopwords.words("english"))
df["message"] = df["message"].apply(lambda x: [word for word in x if word not in stop_words])
print("\n=== AFTER REMOVING STOP WORDS ===")
print(df["message"].head(5))