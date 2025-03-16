import requests
import zipfile
import io

#URL of the dataset
url = "https://archive.ics.uci.edui/static/public/228/sms+spam+collection.zip"
# Download the dataset
response = requests.get(url)
if response.status_code == 200:
    print("Download successful")
else:
    print("Failed to download the dataset")