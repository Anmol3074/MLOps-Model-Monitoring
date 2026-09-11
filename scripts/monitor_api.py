import requests
import pandas as pd

url = "http://127.0.0.1:8000/predict"

# Take one real production sample
df = pd.read_csv("data/production/production_data.csv")
data = df.drop(columns="target").iloc[0].to_dict()
response = requests.post(url, json=data)

print("Status:", response.status_code)
print("Response:", response.text)