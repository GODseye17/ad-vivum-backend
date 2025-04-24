import requests
import json
import sys

term = sys.argv[1]

# 🔹 API Endpoints
SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
DETAILS_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# 🔹 Step 1: Get Article IDs
search_params = {
    "db": "pubmed",
    "term": term,  # Change this search query
    "retmode": "json",
    "retmax": 10,  # Fetch 10 articles
}

response = requests.get(SEARCH_URL, params=search_params)
data = response.json()

article_ids = data.get("esearchresult", {}).get("idlist", [])
print(f"🔹 Found {len(article_ids)} articles.")

# 🔹 Step 2: Fetch Details for Each Article
articles = []
for article_id in article_ids:
    details_params = {
        "db": "pubmed",
        "id": article_id,
        "retmode": "text",
        "rettype": "abstract",  # Retrieve abstract
    }
    details_response = requests.get(DETAILS_URL, params=details_params)

    article_data = {
        "id": article_id,
        "abstract": details_response.text.strip(),
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{article_id}/",
    }
    articles.append(article_data)

# 🔹 Step 3: Save to JSON
with open("datasets/pubmed/scraped_data_pubmed.json", "w") as f:
    json.dump(articles, f, indent=4)

print("✅ Articles saved to scraped_data_pubmed.json")
