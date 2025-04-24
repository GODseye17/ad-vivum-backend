import requests
import json
import os
import time
from dotenv import load_dotenv

# 🔹 Load environment variables
load_dotenv()

API_KEY = os.getenv("SCOPUS_API_KEY")
SEARCH_URL = "https://api.elsevier.com/content/search/scopus"
ABSTRACT_URL = "https://api.elsevier.com/content/abstract/eid/"

query = "TITLE-ABS-KEY(AI in Healthcare)"

params = {
    "query": query,
    "count": 10
}

headers = {
    "X-ELS-APIKey": API_KEY,
    "Accept": "application/json"
}

# 🔹 Fetch initial search results
response = requests.get(SEARCH_URL, params=params, headers=headers)
print("🔍 Search status:", response.status_code)

if response.status_code != 200:
    print("❌ Failed to fetch articles from Scopus.")
    exit()

data = response.json()
entries = data.get("search-results", {}).get("entry", [])

articles = []
for entry in entries:
    eid = entry.get("eid", "")
    if not eid:
        continue

    # 🔹 Fetch full abstract using the Abstract Retrieval API
    abstract_resp = requests.get(
        ABSTRACT_URL + eid,
        headers=headers,
        params={"view": "FULL"}
    )

    abstract = "No abstract available."
    if abstract_resp.status_code == 200:
        try:
            abstract_data = abstract_resp.json()
            coredata = abstract_data.get("abstracts-retrieval-response", {}).get("coredata", {})
            abstract = coredata.get("dc:description", abstract)
        except Exception as e:
            print(f"⚠️ Error parsing abstract for EID {eid}: {e}")
    else:
        print(f"⚠️ Failed to fetch abstract for EID {eid}: {abstract_resp.status_code}")

    articles.append({
        "id": eid,
        "title": entry.get("dc:title", ""),
        "abstract": abstract,
        "url": entry.get("prism:url", f"https://www.scopus.com/record/display.uri?eid={eid}")
    })

    # 💡 Sleep to avoid hitting the API rate limit
    time.sleep(1)

# 🔹 Save results to JSON
os.makedirs("datasets/scopus", exist_ok=True)
with open("datasets/scopus/scraped_data_scopus.json", "w") as f:
    json.dump(articles, f, indent=4)

print(f"✅ Fetched and saved {len(articles)} Scopus articles.")
