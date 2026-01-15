from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from rag_logic import load_pdf, ask_question

app = Flask(__name__)

UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Keep uploaded topics in memory (resets on restart)
UPLOADED_FILES = []


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_files():
    try:
        files = request.files.getlist("pdfs")
        if not files:
            return jsonify({"message": "No files selected", "files": UPLOADED_FILES}), 400

        for file in files:
            if not file.filename:
                continue

            filename = secure_filename(file.filename)
            path = os.path.join(UPLOAD_FOLDER, filename)

            # limit file size (5 MB)
            file.save(path)
            if os.path.getsize(path) > 5 * 1024 * 1024:
                os.remove(path)
                return jsonify({"message": f"{filename} is too large (max 5MB)", "files": UPLOADED_FILES}), 400

            load_pdf(path)

            if filename not in UPLOADED_FILES:
                UPLOADED_FILES.append(filename)

        return jsonify({
            "message": "PDFs uploaded and processed successfully ✅",
            "files": UPLOADED_FILES
        })

    except Exception as e:
        print("UPLOAD ERROR:", e)
        return jsonify({
            "message": "Server error while processing PDFs",
            "files": UPLOADED_FILES
        }), 500


@app.route("/ask", methods=["POST"])
def ask():
    try:
        question = request.form.get("question", "").strip()
        if not question:
            return "Please ask a valid question."

        answer = ask_question(question)
        return answer

    except Exception as e:
        print("ASK ERROR:", e)
        return "Sorry, something went wrong while answering. Please try again."


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
