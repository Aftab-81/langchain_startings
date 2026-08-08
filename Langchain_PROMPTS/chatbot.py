from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "openai/gpt-oss-120b",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

chat_history = [
    SystemMessage(content = "You are a help assistant")
]

while True:
    user_input = input("User: ")
    chat_history.append(HumanMessage(user_input))
    if user_input == "exit":
        break
    chat_history.append(AIMessage(model.invoke(chat_history).content))
    print("AI: ", model.invoke(chat_history).content)

print(chat_history)