from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding_model = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-2", dimensions = 786)

embed = embedding_model.embed_query("Hello")

print(embed)
print(len(embed))
