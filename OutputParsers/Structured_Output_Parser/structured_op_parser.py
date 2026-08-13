from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers.structured import (
    StructuredOutputParser, ResponseSchema
)
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation",
    max_new_tokens = 1024
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name = "Book Name", description = "Name of the Book"),
    ResponseSchema(name = "Author Name", description = "Name of the Author"),
    ResponseSchema(name = "Genre", description = "What's the genre if the book?"),
    ResponseSchema(name = "Publisher Name", description = "Who published this book?"),
    ResponseSchema(name = "Published Year", description = "In which year this book got published?")
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = """
        Give the name, author, genre, publisher name, and published year of the book {book}\n{format_instructions}
    """,
    input_variables = ["book"],
    partial_variables = {"format_instructions": parser.get_format_instructions()}
)

prompt = template.invoke({"book": "Tha Art of Being Alone"})

result = model.invoke(prompt)

parsed_result = parser.parse(result.content)

print(parsed_result)


"""
Advantage of StructuredOutputParser:
It allows us to define predefined fields (schema), making the LLM output
more consistent and predictable.

Disadvantage of StructuredOutputParser:
It defines the expected fields and their descriptions, but it does not
perform strict data-type validation. For example, even if "age" is
described as an integer, the LLM may still return it as a string.
"""