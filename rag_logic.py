import os
import fitz  # pymupdf
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Groq client from env var
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is not set")
client = Groq(api_key=api_key)

# Global storage
documents = []       # list of page texts
sources = []         # list of "filename:page"
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = None  # will be filled after docs are loaded


def _rebuild_index():
    """Rebuild TF-IDF index when new docs are added."""
    global tfidf_matrix
    if not documents:
        tfidf_matrix = None
        return
    tfidf_matrix = vectorizer.fit_transform(documents)


def load_pdf(pdf_path: str):
    """Extract text from PDF and add pages to the index."""
    global documents, sources

    doc = fitz.open(pdf_path)
    base = os.path.basename(pdf_path)

    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if not text:
            continue
        documents.append(text)
        sources.append(f"{base}: page {i+1}")

    _rebuild_index()
    print("Inserted:", pdf_path, "pages:", len(doc))


def load_online_data(url: str):
    """Optional: load website text (if you’re using this)."""
    import requests
    from bs4 import BeautifulSoup

    print("Fetching:", url)
    resp = requests.get(url, timeout=15)
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.extract()

    text = "\n".join(
        line.strip()
        for line in soup.get_text(separator="\n").splitlines()
        if line.strip()
    )
    if not text:
        return

    documents.append(text)
    sources.append(url)
    _rebuild_index()
    print("Inserted online content:", url)


def _retrieve_context(query: str, top_k: int = 3):
    """Return best matching chunks using TF-IDF cosine similarity."""
    if tfidf_matrix is None or not documents:
        return ""

    query_vec = vectorizer.transform([query])
    cosine_similarities = linear_kernel(query_vec, tfidf_matrix).flatten()
    if cosine_similarities.max() == 0:
        return ""

    top_indices = cosine_similarities.argsort()[-top_k:][::-1]
    parts = []
    for idx in top_indices:
        parts.append(f"[{sources[idx]}]\n{documents[idx]}")
    return "\n\n".join(parts)


def ask_question(question: str) -> str:
    """Use Groq model + retrieved context to answer."""
    context = _retrieve_context(question)

    prompt = f"""
Answer in 3–5 short bullet points.
Use ONLY this context if it is relevant:

{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="gemma2-9b-it",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
