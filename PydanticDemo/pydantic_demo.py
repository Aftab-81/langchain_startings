from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from pydantic import BaseModel
load_dotenv()

"""
    Here the code is not working reason is we used HF model instead use OpenAI model
    for structured output. The model we are using is not trained for structured output.
"""

class Student(BaseModel):
    name: str
    age:int
    course: str


llm = HuggingFaceEndpoint(
    repo_id = "openai/gpt-oss-120b",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

structured_model = model.with_structured_output(Student)

prompt = """
    Extract the student's information from this text:

    Aftab is 21 years old and is studying Computer Science
    with a specialization in Artificial Intelligence and Machine Learning.
"""

response = structured_model.invoke(prompt)
print(response)





