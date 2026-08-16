from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnableBranch

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "openai/gpt-oss-120b",
    task = "text-generation",
    max_new_tokens = 200
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template = "generate a detiled report on {topic}",
    input_variables = ["topic"]
)

template2 = PromptTemplate(
    template = "Give a 3-5 line summary of following: {text}",
    input_variables = ["text"]
)

parser = StrOutputParser()

sequence_chain = RunnableSequence(template1, model, parser)

runnable_branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 100, RunnableSequence(template2, model, parser)),
    (lambda x: len(x.split()) < 100, RunnablePassthrough()),
    RunnableLambda(lambda x: "Model has generate exact amount of tokens")
)

final_chain = RunnableSequence(sequence_chain, runnable_branch_chain)

result = final_chain.invoke({"topic": "Football"})

print(result)