import faiss
import numpy as np
import json
import google.generativeai as genai  # Google Gemini
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 🔹 Load FAISS index
index = faiss.read_index("news_index.faiss")

# 🔹 Load articles
with open("scraped_data.json", "r") as f:
    articles = json.load(f)

# 🔹 Initialize Google Gemini API (replace YOUR_API_KEY)
genai.configure(api_key="AIzaSyA-AfbLuDw6cJbWkU3w8ADhNfXj6DGEQ0Y")
model = genai.GenerativeModel("gemini-1.5-flash")

# 🔹 Function to retrieve and answer queries
def answer_query(query):
    query_vector = embeddings.embed_query(query)
    _, indices = index.search(np.array([query_vector], dtype=np.float32), k=3)  # Top 3 articles

    retrieved_articles = [articles[i] for i in indices[0] if i < len(articles)]
    if not retrieved_articles:
        return "⚠️ No relevant articles found."

    context = "\n\n".join([f"{article['abstract']}" for article in retrieved_articles])
    prompt = f"Based on the following PubMed abstracts, answer this question:\n\n{context}\n\nQ: {query}\nA:"

    # 🔹 Generate response using Gemini
    response = model.generate_content(prompt)
    return response.text

# 🔹 Example usage
query = "Why is breast cancer caused?"
print(answer_query(query))
