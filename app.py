from flask import Flask, jsonify, request, render_template
from qa_retriever import QARetriver

app = Flask(__name__)
app.config["BOOTSTRAP_SERVE_LOCAL"] = True

qa = QARetriver()


@app.route("/admin", methods=["GET", "POST"])
def generate():
    if request.method == "POST":
        file_path = request.form["file_path"]
        embedding_index = request.form["embedding_index"]

        try:
            qa.generate_embeddings(
                src_file_path=file_path, embeddings_index=embedding_index
            )

            return render_template(
                "generate_result.html",
                result_message=f"Generated embeddings for {file_path} and saved with index name as {embedding_index}",
            )

        except Exception as e:
            return jsonify({"error": str(e)})

    return render_template("admin.html")


@app.route("/", methods=["GET", "POST"])
def ask_query():
    if request.method == "POST":
        query = request.form["query"]

        try:
            resp = qa.retrieve(query=query)

            return render_template("result.html", response=resp)

        except Exception as e:
            return jsonify({"error": str(e)})

    return render_template("query.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
