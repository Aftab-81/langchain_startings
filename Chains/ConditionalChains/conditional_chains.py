from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableBranch, RunnableParallel, RunnableLambda
from typing import Literal
from dotenv import load_dotenv

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation",
    max_new_tokens = 1024
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description = "Give the sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object = Feedback)


prompt1 = PromptTemplate(
    template = "Classify the sentiment of the following feedback text either into Positive or Negative\n{feedback}\n{format_instructions}",
    input_variables = ["feedback"],
    partial_variables = {"format_instructions": parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template = "Write an appropriate response to this positive feedback \n{feedback}",
    input_variables = ["feedback"]
)

prompt3 = PromptTemplate(
    template = "Write an appropriate apology reply to this negative feedback \n{feedback}",
    input_variables = ["feedback"]
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x: x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "Could not find sentiment")
)

final_chain = classifier_chain | branch_chain

result = final_chain.invoke({"feedback": "The newly bought phone is defective. What a shame. I wasted my 12000 dollars"})
print(classifier_chain.invoke({"feedback": "The newly bought phone is defective. What a shame. I wasted my 12000 dollars"}).sentiment)
print("\n")
print("*" * 60)

print(result)

final_chain.get_graph().print_ascii()