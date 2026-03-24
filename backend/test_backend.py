import requests

url = "http://127.0.0.1:5000/upload"

with open("raw_with_anomalies.csv", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

print(response.json())