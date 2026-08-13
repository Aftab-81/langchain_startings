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

template = PromptTemplate(
    template = "Give me the details about DongHua {donghua}",
    input_variables = ["donghua"]
)

chain = template | model | parser

result = chain.invoke({"donghua": "Battle Through The Heavens"})

print(result)

print("\n")

chain.get_graph().print_ascii()

"""
    input(donghua) -> prompt template -> LLM -> output parser -> output.
"""