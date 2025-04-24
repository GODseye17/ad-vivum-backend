import requests

API_KEY = "d41f95be2b2e9c6b76308f6fd517c5ed"
EID = "2-s2.0-85070359767"  # example EID, use one from your list

headers = {
    "X-ELS-APIKey": API_KEY,
    "Accept": "application/json"
}

url = f"https://api.elsevier.com/content/abstract/eid/{EID}"

response = requests.get(url, headers=headers, params={"view": "FULL"})

print("Status Code:", response.status_code)
print(response.json())
