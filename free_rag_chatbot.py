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

# Create persistent DB folder
chroma_client = PersistentClient(path="./pdf_db")
collection = chroma_client.get_or_create_collection(name="pdf_data")


# Load PDF into DB
def load_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    print("\nExtracting PDF text...")
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            emb = embed_model.encode([text])[0].tolist()
            collection.add(
                ids=[f"page_{i}"],
                documents=[text],
                embeddings=[emb]
            )
    print("PDF inserted into vector DB!")


# Ask chatbot
def ask_question(question):
    query_emb = embed_model.encode([question])[0].tolist()

    result = collection.query(
        query_embeddings=[query_emb],
        n_results=2
    )

    context = "\n".join(result['documents'][0])

    prompt = f"""
    Use the below PDF content to answer accurately:

    {context}

    Question: {question}
    """

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# ---------------- RUN ----------------

load_pdf("uploads/Python_Notes.pdf")

print("\nChatbot Ready! Ask your questions!")

while True:
    q = input("\nAsk: ")
    if q.lower() == "exit":
        break

    answer = ask_question(q)
    print("\n💡 Answer:", answer)
