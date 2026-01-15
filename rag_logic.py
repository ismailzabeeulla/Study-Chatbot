import os
import fitz  # PyMuPDF
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Groq client
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY not set")

client = Groq(api_key=api_key)

# In-memory store
documents = []
sources = []

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = None


def rebuild_index():
    global tfidf_matrix
    if documents:
        tfidf_matrix = vectorizer.fit_transform(documents)


def load_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    name = os.path.basename(pdf_path)

    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            documents.append(text)
            sources.append(f"{name} (page {i+1})")

    rebuild_index()
    print("Loaded:", name)


def retrieve_context(query, top_k=3):
    if not tfidf_matrix:
        return ""

    q_vec = vectorizer.transform([query])
    sims = linear_kernel(q_vec, tfidf_matrix).flatten()

    if sims.max() == 0:
        return ""

    top_idx = sims.argsort()[-top_k:][::-1]
    return "\n\n".join(
        f"[{sources[i]}]\n{documents[i]}" for i in top_idx
    )


def ask_question(question):
    context = retrieve_context(question)

    prompt = f"""
Answer in simple language using 3–5 bullet points.
Use the context only if it helps.

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",  # ensure active model
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content
