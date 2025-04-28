import faiss
import numpy as np
import json
import subprocess
import google.generativeai as genai  # Google Gemini
import requests
from fastapi import HTTPException

#from langchain_community.embeddings import HuggingFaceEmbeddings

#from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer


embeddings = SentenceTransformer("models/all-MiniLM-L6-v2")

# 🔹 Initialize Google Gemini API (replace YOUR_API_KEY)
genai.configure(api_key="AIzaSyA-AfbLuDw6cJbWkU3w8ADhNfXj6DGEQ0Y")
model = genai.GenerativeModel("gemini-1.5-flash")

def fetch_pubmed_articles(term):
    # 🔹 API Endpoints
    SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    DETAILS_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    # 🔹 Step 1: Get Article IDs
    search_params = {
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retmax": 6,
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
            "rettype": "abstract",
        }
        details_response = requests.get(DETAILS_URL, params=details_params)

        article_data = {
            "id": article_id,
            "abstract": details_response.text.strip(),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{article_id}/",
        }
        articles.append(article_data)

    return articles

# def load_dataset(name):
#     try:
#         index = faiss.read_index(f"datasets/{name}/news_index_{name}.faiss")
#         with open(f"datasets/{name}/scraped_data_{name}.json", "r") as f:
#             articles = json.load(f)
#         return index, articles
#     except Exception as e:
#         print(f"⚠️ Could not load {name} dataset:", e)
#         return None, []

# 🔹 Query Handler
def answer_query(query, source="both"):
    print("Fetch fresh articles for:", query)
    articles = fetch_pubmed_articles(query)
    abstracts = [article["abstract"] for article in articles]
    
    if not abstracts:
        raise HTTPException(status_code=400, detail="No articles found.")
    
    vectors = embeddings.encode(abstracts)
    print(f"Generated vectors for {len(abstracts)} abstracts.")

    d = vectors.shape[1]
    print(f"Vector dimensionality: {d}")

    if d == 0:
        raise HTTPException(status_code=500, detail="Vector dimensionality is 0. Check your embedding model.")

    index = faiss.IndexFlatL2(d)
    index.add(np.array(vectors).astype(np.float32))
    
    query_vector = embeddings.encode([query])[0].astype(np.float32)
    _, I = index.search(np.array([query_vector]), k=min(3, len(abstracts)))

    # Ensure that there are enough articles to return
    if len(I[0]) == 0:
        raise HTTPException(status_code=500, detail="No results found in FAISS search.")

    selected_abstracts = [abstracts[i] for i in I[0]]
    context = "\n\n".join(selected_abstracts)
    prompt = f"Based on the following abstracts, answer this question:\n\n{context}\n\nQ: {query}\nA:"

    response = model.generate_content(prompt)
    return response.text


# 🔹 CLI Usage
if __name__ == "__main__":
    #print("Choose dataset: [pubmed / scopus / both]")
    selected = "pubmed"

    query = input("Enter your query:\n> ")
    print("\n📚 Response:\n")
    print(answer_query(query, selected))