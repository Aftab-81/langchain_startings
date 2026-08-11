from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import BaseModel
from pydantic import Field
from typing import Annotated
from typing import Optional
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "LiquidAI/LFM2.5-2.6B",
    task = "text-generation",
    max_new_tokens = 512 # Without this Uncomplete string error occurs.
)

model = ChatHuggingFace(llm=llm)

class Review(BaseModel):
    key_themes: list[str] = Field(description = "Mention all the key themes discussed in the review")
    # Annotated adds additional metadata/instructions to the field.
    # The first argument (str) defines the data type.
    # The second argument describes what the LLM should generate for this field.
    summary: str = Field(description = "Give the brief summary of the review")
    sentiment: Literal["pos", "neg", "neu"] = Field(description = "Classify the sentiment of the review either")
    pros: Optional[list[str]] = Field(description = "List down all the pros mentioned in the review. If None, return None")
    cons: Optional[list[str]] = Field(description = "List down all the cons mentioned in the review. If None, return None")
    name: Optional[str] = Field(description = "Write the name of the reviewer if mentioned in the review")


# LangChain uses the Review schema in the background to tell the LLM
# what structure the output should follow. The schema/instructions are
# automatically attached to the model request; we don't manually write
# them inside the prompt.   

"""
    # LangChain uses the Review Pydantic schema in the background
# to tell the LLM what structure the output should follow.
#
# However, ChatHuggingFace's function-calling implementation
# in this setup does NOT support Pydantic schemas.
#
# Therefore, the following line raises:
#
# NotImplementedError:
# Pydantic schema is not supported for function calling
#
# To properly execute this same Pydantic structured-output code,
# use an OpenAI chat model such as ChatOpenAI instead of
# ChatHuggingFace.
#
# Example:
#
# from langchain_openai import ChatOpenAI
#
# model = ChatOpenAI(
#     model="gpt-4o-mini",
#     temperature=0
# )
#
# structured_model = model.with_structured_output(Review)
#
# The rest of the code can remain the same.

"""
structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""
    These headphones have both impressed and disappointed me. On the positive side, they are very lightweight and 
    easy to carry around. The battery is excellent, and I can use them for several days without charging them. 
    I also like that they connect quickly to my phone and laptop. For podcasts, YouTube videos, and online classes, 
    the audio quality is more than sufficient. However, when I listen to music, I feel that the sound could be more detailed. 
    The bass is decent, but some songs don't sound as spacious or immersive as they do on more expensive headphones. 
    Another issue is the on-ear design. After extended use, my ears become warm and slightly painful. 
    There is also no ANC, which makes them less useful when travelling. 
    I think these are good headphones for someone who wants something simple and affordable, 
    but I wouldn't recommend them to someone who is particularly concerned about audio quality or long-term comfort.

    Reviewed by Aftabalam Makandar
""")


print(type(result))
print(result.summary)
print(result.sentiment)
print(result.name)