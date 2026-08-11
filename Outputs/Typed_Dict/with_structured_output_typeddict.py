from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "openai/gpt-oss-120b",
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm)

class Review(TypedDict):
    summary: str
    sentiment: str


# LangChain uses the Review schema in the background to tell the LLM
# what structure the output should follow. The schema/instructions are
# automatically attached to the model request; we don't manually write
# them inside the prompt.   

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""
    These headphones have both impressed and disappointed me. On the positive side, they are very lightweight and easy to carry around. The battery is excellent, and I can use them for several days without charging them. I also like that they connect quickly to my phone and laptop. For podcasts, YouTube videos, and online classes, the audio quality is more than sufficient. However, when I listen to music, I feel that the sound could be more detailed. The bass is decent, but some songs don't sound as spacious or immersive as they do on more expensive headphones. Another issue is the on-ear design. After extended use, my ears become warm and slightly painful. There is also no ANC, which makes them less useful when travelling. I think these are good headphones for someone who wants something simple and affordable, but I wouldn't recommend them to someone who is particularly concerned about audio quality or long-term comfort.
""")


print(type(result))
print(result["summary"])
print(result["sentiment"])