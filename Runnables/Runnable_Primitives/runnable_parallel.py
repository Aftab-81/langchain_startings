from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "openai/gpt-oss-120b",
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template = "Write a detailed report on {topic}",
    input_variables = ["topic"]
)

template2 = PromptTemplate(
    template = "Write a important questions on text: {topic}",
    input_variables = ["topic"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        "tweeter": RunnableSequence(template1, model, parser),
        "linked_in": RunnableSequence(template2, model, parser)
    }
)

result = parallel_chain.invoke({"topic": "ML"})
print(result["tweeter"])
print("\n", result["linked_in"])