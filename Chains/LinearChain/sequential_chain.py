from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation",
    max_new_tokens = 1024
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = "Give me a detailed report on topic {topic}",
    input_variables = ["topic"]
)

prompt2 = PromptTemplate(
    template = "Give me a 5 points summary on the text: {text}",
    input_variables = ["text"]
)

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"topic": "Battle Through The Heavens"})

print(result)

chain.get_graph().print_ascii()