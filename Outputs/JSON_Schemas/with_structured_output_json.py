from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import BaseModel
from pydantic import Field
from typing import Annotated
from typing import Optional
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "openai/gpt-oss-120b",
    task = "text-generation",
    max_new_tokens = 512 # Without this Uncomplete string error occurs.
)

model = ChatHuggingFace(llm=llm)

# schema

json_schema = {
    "title": "Review",
    "type": "object",
    "properties": {
        "key_themes": {
            "type": "array",
            "items":{
                "type": "string"
            },
            "description": "Mention all the key themes discussed in the review"
        },

        "summary": {
            "type": "string",
            "description": "Give the brief summary of the review"
        },

        "sentiment": {
            "type": "string",
            "enum": ["pos", "neg", "neu"],
            "description": "Classify the sentiment of the review either Positive, Negative or Neutral"
        },

        "pros": {
            "type": ["array", "null"],
            "items": {
                "type": "string"
            },
            "description": "List down all the pros mentioned in the review. If None, return None"
        },

        "cons": {
            "type": ["array", "null"],
            "items": {
                "type": "string"
            },
            "description": "List down all the cons mentioned in the review. If None, return None"
        },

        "name": {
            "type": ["string", "null"],
            "description": "Write the name of the reviewer if mentioned in the review"
        }
    },
    "required": ["key_themes", "summary", "sentiment"]
}


# LangChain uses the Review schema in the background to tell the LLM
# what structure the output should follow. The schema/instructions are
# automatically attached to the model request; we don't manually write
# them inside the prompt.   

structured_model = model.with_structured_output(json_schema)

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

print(result)
print("*" * 60)
print(type(result))
print("*" * 60)
print(result["summary"])
print(result["sentiment"])
print(result["name"])