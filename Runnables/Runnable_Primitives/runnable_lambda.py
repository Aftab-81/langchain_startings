from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda

load_dotenv()

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
    template = "Explain in easy following: {topic_of_joke}",
    input_variables = ["topic_of_joke"]
)

parser = StrOutputParser()

sequence_chain = RunnableSequence(template1, model, parser)

parallel_chain = RunnableParallel(
    {
        "joke": RunnablePassthrough(),
        "count": RunnableLambda(lambda x: len(x.split())),
        "explanation": RunnableSequence(template2, model, parser)
    }
)

final_chain = RunnableSequence(sequence_chain, parallel_chain)

result = final_chain.invoke({"topic_of_joke": "ML Secrets"})

print(result)