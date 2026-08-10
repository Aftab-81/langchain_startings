from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

chat_template = ChatPromptTemplate(
    [
        ("system", "You are master in {domain}"),
        ("human", "Explain the topic {topic} in very simple terms")
    ]
)

prompt = chat_template.invoke(
    {
        "domain": "Python programming",
        "topic": "Decorators"
    }
)

print(prompt)