import faiss
import numpy as np
import json
import subprocess
import google.generativeai as genai  # Google Gemini
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 🔹 Initialize Google Gemini API (replace YOUR_API_KEY)
genai.configure(api_key="AIzaSyA-AfbLuDw6cJbWkU3w8ADhNfXj6DGEQ0Y")
model = genai.GenerativeModel("gemini-1.5-flash")

def load_dataset(name):
    try:
        index = faiss.read_index(f"datasets/{name}/news_index_{name}.faiss")
        with open(f"datasets/{name}/scraped_data_{name}.json", "r") as f:
            articles = json.load(f)
        return index, articles
    except Exception as e:
        print(f"⚠️ Could not load {name} dataset:", e)
        return None, []

# 🔹 Query Handler
def answer_query(query, source="both"):
    print("Fetch fresh articles for:", query)
    subprocess.run(["python3", "datasets/pubmed/fetch_pubmed.py", query], check=True)
    subprocess.run(["python3", "datasets/pubmed/store_faiss_pubmed.py"], check=True)

    k = 3
    query_vector = embeddings.embed_query(query)
    all_articles = []

    if source in ["pubmed", "both"]:
        pubmed_index, pubmed_articles = load_dataset("pubmed")
        if pubmed_index:
            _, indices = pubmed_index.search(np.array([query_vector], dtype=np.float32), k)
            all_articles += [pubmed_articles[i] for i in indices[0] if i < len(pubmed_articles)]

    if source in ["scopus", "both"]:
        scopus_index, scopus_articles = load_dataset("scopus")
        if scopus_index:
            _, indices = scopus_index.search(np.array([query_vector], dtype=np.float32), k)
            all_articles += [scopus_articles[i] for i in indices[0] if i < len(scopus_articles)]

    if not all_articles:
        return "⚠️ No relevant articles found."

    context = "\n\n".join([article["abstract"] for article in all_articles])
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