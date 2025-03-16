import requests
import zipfile
import io
import os
import pandas as pd

#URL of the dataset
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
# Download the dataset
response = requests.get(url)
if response.status_code == 200:
    print("Download successful")
else:
    print("Failed to download the dataset")

# extract the dataset
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    z.extractall("sms_spam_collection")
    print("Extraction successful")

# list extracted files
extracted_files = os.listdir("sms_spam_collection")
print("Extracted files:", extracted_files)

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