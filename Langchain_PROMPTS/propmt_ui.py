from tempfile import template

from flask import Flask, render_template, request
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate, load_prompt

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
        paper_input = request.form["paper_input"]
        style_input = request.form["style_input"]
        length_input = request.form["length_input"]

        template = load_prompt("Langchain_PROMPTS\prompt_template.json")


        try:
            chain = template | model

            result = chain.invoke({     # Langchain Chain Concept Both template and model required invoke method to execute
            "paper_input": paper_input, 
            "style_input": style_input,
            "length_input": length_input
        })
            
            response = result.content
        except Exception as e:
            response = str(e)

    return render_template("index.html", response=response)

if __name__ == "__main__":
    app.run(host = "localhost", port = 1234, debug=True)