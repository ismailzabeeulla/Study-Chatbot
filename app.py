from flask import Flask, render_template, request
import os
from rag_logic import load_pdf, load_online_data, ask_question


app = Flask(__name__)
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    files = request.files.getlist("pdfs")
    for file in files:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        load_pdf(file_path)
    return "PDFs uploaded and processed successfully!"


@app.route("/ask", methods=["POST"])
def chat():
    data = request.form["question"]
    answer = ask_question(data)
    return answer



if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


