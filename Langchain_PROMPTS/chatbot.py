from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "openai/gpt-oss-120b",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

chat_history = []

while True:
    user_input = input("User: ")
    chat_history.append(user_input)
    if user_input == "exit":
        break
    chat_history.append(model.invoke(chat_history).content)
    print("AI: ", model.invoke(chat_history).content)

print(chat_history)