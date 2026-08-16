from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda

load_dotenv()

## LCER: LangChain Expression Language

"""
    Instead of using RunnableSequence class in order to create a chain use | operator.
    | operator is jsut an alternative to RunnableSequence class.
"""

llm = HuggingFaceEndpoint(
    repo_id = "openai/gpt-oss-120b",
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template = "Write a joke on {topic_of_joke}",
    input_variables = ["topic_of_joke"]
)

template2 = PromptTemplate(
    template = "Give a 4-6 lines of Summary to the following: {text}",
    input_variables = ["text"]
)

parser = StrOutputParser()

chain1 = template1 | model | parser 
chain2 = template2 | model | parser

final_chain = chain1 | chain2

result = final_chain.invoke({"topic_of_joke": "Hawaii"})

print(result)