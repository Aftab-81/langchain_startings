from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text = "The capital of India is New Delhi"

embedding = embeddings.embed_query(text)

print(len(embedding))   # 384
print(embedding[:10])   # First 10 values