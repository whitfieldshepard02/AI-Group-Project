# =========================
# FULL RAG WITH OLLAMA (ENGINEERING WIKIPEDIA)
# =========================

import wikipedia
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer

# -------------------------
# CONFIG
# -------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"   # change if you use mistral, etc.

# -------------------------
# STEP 1: ENGINEERING TOPICS
# -------------------------

engineering_topics = [
    "Mechanical engineering",
    "Electrical engineering",
    "Civil engineering",
    "Chemical engineering",
    "Aerospace engineering",
    "Biomedical engineering",
    "Industrial engineering",
    "Software engineering",
    "Materials science",
    "Thermodynamics",
    "Fluid mechanics",
    "Control systems",
    "Signal processing",
    "Structural engineering",
    "Robotics",
    "Heat transfer",
    "Manufacturing engineering",
]

def fetch_articles(topics, max_articles_per_topic=8):
    articles = []

    for topic in topics:
        try:
            results = wikipedia.search(topic, results=max_articles_per_topic)

            for title in results:
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    articles.append(page.content)
                    print(f"Fetched: {title}")
                except:
                    continue
        except:
            continue

    return articles


# -------------------------
# STEP 2: CHUNKING
# -------------------------

def chunk_text(text, chunk_size=300):
    words = text.split()
    return [
        " ".join(words[i:i+chunk_size])
        for i in range(0, len(words), chunk_size)
    ]


# -------------------------
# STEP 3: BUILD DATASET
# -------------------------

print("\nCollecting engineering articles...\n")
articles = fetch_articles(engineering_topics)

all_chunks = []
for article in articles:
    all_chunks.extend(chunk_text(article))

print(f"\nTotal chunks: {len(all_chunks)}")


# -------------------------
# STEP 4: EMBEDDINGS
# -------------------------

print("\nGenerating embeddings...\n")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = embed_model.encode(all_chunks, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")


# -------------------------
# STEP 5: FAISS INDEX
# -------------------------

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(f"FAISS index built with {index.ntotal} vectors")


# -------------------------
# STEP 6: RETRIEVER
# -------------------------

def retrieve(query, top_k=5):
    q_vec = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(q_vec, top_k)

    return [all_chunks[i] for i in indices[0]]


# -------------------------
# STEP 7: LLM (OLLAMA)
# -------------------------

def ask_ollama(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]


# -------------------------
# STEP 8: FULL RAG PIPELINE
# -------------------------

def rag_query(question):
    contexts = retrieve(question, top_k=5)

    context_text = "\n\n---\n\n".join(contexts)

    prompt = f"""
You are an engineering expert.

Answer the question ONLY using the context below.
If the answer is not contained, say "I don't know".

Context:
{context_text}

Question:
{question}

Answer:
"""

    return ask_ollama(prompt)


# -------------------------
# STEP 9: INTERFACE
# -------------------------

while True:
    q = input("\nAsk an engineering question (or 'exit'): ")

    if q.lower() == "exit":
        break

    answer = rag_query(q)

    print("\n=== ANSWER ===\n")
    print(answer)