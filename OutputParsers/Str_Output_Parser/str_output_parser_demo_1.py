from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from config.configuration import MODEL_NAME
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id=MODEL_NAME,
    task="text-generation",
    max_new_tokens = 1024
)

model = ChatHuggingFace(llm=llm)


template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)


template2 = PromptTemplate(
    template="Write a brief summary on the following text\n{report}",
    input_variables=["report"]
)

# Create a string output parser to convert the model's response
# into a plain Python string.
parser = StrOutputParser()

# Create a sequential chain using the pipe (|) operator.
# Prompt 1 generates a detailed report, which is passed to the model.
# The parser converts the model response into a string.
# That string is then passed to Prompt 2 as the {report} value.
# Finally, the second model response is parsed into a string
chain = template1 | model | parser | template2 | model | parser

# Invoke the complete chain by providing the value for {topic}.
result = chain.invoke({"topic": "Deep Sea"})

print(result)
