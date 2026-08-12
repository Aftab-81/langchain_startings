from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation",
    max_new_tokens = 1024
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template = "Give me the story of Jujutsu Kaisen Main hero & show stellers information.\n{format_instructions}",
    input_variables=[],
    partial_variables = {"format_instructions": parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({})

print(result)   

# Disadvantage:
# We have not defined a fixed structure/schema for the JSON output.
# So the LLM can decide which fields to return.
# The output may be different each time.