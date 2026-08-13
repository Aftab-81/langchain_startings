from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation",
    max_new_tokens = 1024
)

model = ChatHuggingFace(llm=llm)

class Book(BaseModel):
    name: str = Field(description = "Name of the Book")
    author: str = Field(description = "Name of the Author")
    genre: str = Field(description = "What's the genre if the book?")
    publisher_name: str = Field(description = "Who published this book?")
    published_yaer: int = Field(description = "In which year this book got published?")

parser = PydanticOutputParser(pydantic_object = Book)

template = PromptTemplate(
    template = """
        Give the name, author, genre, publisher name, and published year of the book {book}\n{format_instructions}
    """,
    input_variables = ["book"],
    partial_variables = {"format_instructions": parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({"book": "How to read people like a book"})

print(result)

print("\n\n")
print("*" * 50)

print(template.invoke({"book": "How to read people like a book"}))