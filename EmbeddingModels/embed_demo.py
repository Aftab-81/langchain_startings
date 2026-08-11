from sentence_transformers import SentenceTransformer

model = SentenceTransformer("perplexity-ai/pplx-embed-v1-4b")

text = "LangChain is a framework for developing applications powered by language models."

embed = model.encode(text)

print(embed)