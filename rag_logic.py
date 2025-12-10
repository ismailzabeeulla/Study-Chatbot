import fitz
import os
from groq import Groq
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient




api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is not set. Please set it as an environment variable.")

client = Groq(api_key=api_key)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = PersistentClient(path="./pdf_db")
collection = chroma_client.get_or_create_collection(name="pdf_data")


def load_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            emb = embed_model.encode([text])[0].tolist()
            collection.add(
                ids=[f"{pdf_path}_page_{i}"],
                documents=[text],
                embeddings=[emb]
            )
    print("Inserted:", pdf_path)


def ask_question(question):
    query_emb = embed_model.encode([question])[0].tolist()

    result = collection.query(
        query_embeddings=[query_emb],
        n_results=2
    )

    context = "\n".join(result['documents'][0])

    prompt = f"""
    Answer strictly in simple bullet points.
    Make answer short, clear and easy.
    Do NOT write long paragraphs.
    Use only the PDF content below:

    Context:
    {context}

    Question:
    {question}

    Your answer format:
    - point 1
    - point 2
    - point 3
    """

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
