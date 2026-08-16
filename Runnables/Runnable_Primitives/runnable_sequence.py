from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation",
    max_new_tokens = 1024
)

model = ChatHuggingFace(llm=llm)

template = PromptTemplate(
    template = "Write a joke on {topic}",
    input_variables = ["topic"]
)

template2 = PromptTemplate(
    template = "Give me the actual meanning of the text: {response}",
    input_variables = ["response"]
)

parser = StrOutputParser()

chain = RunnableSequence(
    template,
    model,
    parser,
    template2,
    model,
    parser
)

result = chain.invoke({"topic": "Anime"})

print(result)