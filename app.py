from flask import Flask, render_template, request, jsonify  # add jsonify
import os
from rag_logic import load_pdf, load_online_data, ask_question

app = Flask(__name__)
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

UPLOADED_FILES = []  # keep track of uploaded PDFs


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    from werkzeug.utils import secure_filename
    files = request.files.getlist("pdfs")
    new_files = []

    for file in files:
        if not file.filename:
            continue
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        load_pdf(file_path)
        new_files.append(filename)
        if filename not in UPLOADED_FILES:
            UPLOADED_FILES.append(filename)

    return jsonify({
        "message": "PDFs uploaded and processed ✅",
        "files": UPLOADED_FILES
    })


@app.route("/ask", methods=["POST"])
def chat():
    data = request.form["question"]
    answer = ask_question(data)
    return answer



if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


