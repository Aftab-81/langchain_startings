from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "openai/gpt-oss-120b",
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template = "Write a joke on {topic}",
    input_variables = ["topic"]
)

template2 = PromptTemplate(
    template = "Summarize this text in easy language: {topic}",
    input_variables = ["topic"]
)

parser = StrOutputParser()

sequence_chain = RunnableSequence(template1, model, parser)

parallel_chain = RunnableParallel(
    {
        "joke":  RunnablePassthrough(),
        "explanation": RunnableSequence(template2, model, parser)
    }
)

final_chain = RunnableSequence(sequence_chain, parallel_chain)

result = final_chain.invoke({"topic": "Pikachu"})
print(result)