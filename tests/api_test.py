import requests

response = requests.post(
    "http://localhost:8000/api/ask",
    json={"question": "What information do you have about label leaking?"}
)
print(response.json())