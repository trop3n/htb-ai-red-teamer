import requests
import zipfile
import io
import os

#URL of the dataset
url = "https://archive.ics.uci.edui/static/public/228/sms+spam+collection.zip"
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