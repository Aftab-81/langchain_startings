from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation",
    max_new_tokens = 1024
)

model = ChatHuggingFace(llm=llm)

template = PromptTemplate(
    template = "Summarize the following text: {question}",
    input_variables = ["question"]
)

url = "https://en.wikipedia.org/wiki/Battle_Through_the_Heavens"

loader = WebBaseLoader(web_path = url)

docs = loader.load()

parser = StrOutputParser()

chain = template | model | parser

print(chain.invoke({
    "question": docs[0].page_content
}))