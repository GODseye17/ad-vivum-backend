import faiss
import json
import numpy as np
import os
import sys
from langchain.embeddings import HuggingFaceEmbeddings

#from utils.embeddings import get_embedding_model

# 🔹 Load Scopus metadata
with open("datasets/scopus/scraped_data_scopus.json", "r") as f:
    articles = json.load(f)

# 🔹 Get abstracts
article_texts = [article["abstract"] for article in articles if article["abstract"]]

if not article_texts:
    print("⚠️ No abstracts found.")
    exit()

# 🔹 Embed using HuggingFace
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

article_vectors = embeddings.embed_documents(article_texts)

# 🔹 Create FAISS index
dimension = len(article_vectors[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(article_vectors, dtype=np.float32))

# 🔹 Save index
faiss.write_index(index, "datasets/scopus/news_index_scopus.faiss")
print("✅ Scopus FAISS index created.")
