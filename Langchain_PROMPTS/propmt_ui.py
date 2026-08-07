from flask import Flask, render_template, request
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

app = Flask(__name__)

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

@app.route("/", methods=["GET", "POST"])
def index():
    response = ""

    if request.method == "POST":
        prompt = request.form["prompt"]

        try:
            result = model.invoke(prompt)
            response = result.content
        except Exception as e:
            response = str(e)

    return render_template("index.html", response=response)

if __name__ == "__main__":
    app.run(debug=True)