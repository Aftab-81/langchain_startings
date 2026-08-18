from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embed = GoogleGenerativeAIEmbeddings(
    model = "gemini-embedding-2",
    dimensions = 100
)

splitter = SemanticChunker(
    embeddings = embed,
    breakpoint_threshold_amount = 1,
    breakpoint_threshold_type = "standard_deviation"
)

text = """
    Farmers were working hard in the fields, preparing the soil and planting
seeds for the next season. The sun was bright, and the air smelled of
earth and fresh grass.The Indian Premier League (IPL) is the biggest cricket league in the world.
People all over the world watch the matches and cheer for their favourite
teams.

Terrorism is a big danger to peace and safety. It causes harm to people and
creates fear in cities and villages. When such attacks happen, they leave
behind pain and sadness. To fight terrorism, we need strong laws, alert
security forces, and support from people who care about peace and safety.

"""

chunks = splitter.create_documents([text]) # create_documents() return a list of Documnet objects

print(chunks[0])
print("\n")
print(chunks[1])
print("\n")
print(chunks[2]) # Returns page_content & metadata
print(len(chunks))

print(type(chunks[0]))  # <class 'langchain_core.documents.base.Document'>

print("\n")
print(chunks)