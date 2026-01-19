import requests
import json

# Test translation endpoint
url = "http://localhost:8000/api/translate"
data = {
    "text": "The patient has a fever and headache",
    "source": "en",
    "target": "vi"
}

print("Testing translation API...")
print(f"Input: {data['text']}")
print()

response = requests.post(url, json=data)
print(f"Status Code: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
