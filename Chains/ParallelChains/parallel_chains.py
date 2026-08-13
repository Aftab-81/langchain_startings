from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()


llm1 = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation",
    max_new_tokens = 1024
)

model1 = ChatHuggingFace(llm = llm1)

llm2 = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    max_new_tokens = 1024
)

model2 = ChatHuggingFace(llm = llm2)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = "Generate a short and simple notes from the topic {topic}",
    input_variables = ["topic"]
)

prompt2 = PromptTemplate(
    template = "Generate 5 short question-answers from the topic {topic}",
    input_variables = ["topic"]
)

prompt3 = PromptTemplate(
    template = "Merge the provided notes and quiz into a document\n {notes} and {quiz}",
    input_variables = ["notes", "quiz"]
)

parallel_chain = RunnableParallel(
    {
        "notes": prompt1 | model1 | parser,
        "quiz": prompt2 | model2 | parser
    }
)

merge_chain = prompt3 | model1 | parser

final_chain = parallel_chain | merge_chain

result = final_chain.invoke({"topic": "Machine Learning"})

print(result)

final_chain.get_graph().print_ascii()







