import faiss
import json
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings  # Use Hugging Face instead of OpenAI

# 🔹 Load scraped data
with open("datasets/pubmed/scraped_data_pubmed.json", "r") as f:
    articles = json.load(f)

# 🔹 Initialize Hugging Face Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 🔹 Convert articles to embeddings
article_texts = [article["abstract"] for article in articles if article["abstract"]]
if not article_texts:
    print("⚠️ No valid articles found! Exiting...")
    exit()

article_vectors = embeddings.embed_documents(article_texts)

# 🔹 Store vectors in FAISS
dimension = len(article_vectors[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(article_vectors, dtype=np.float32))

# 🔹 Save FAISS index
faiss.write_index(index, "datasets/pubmed/news_index_pubmed.faiss")
print("✅ FAISS index saved as news_index.faiss")
