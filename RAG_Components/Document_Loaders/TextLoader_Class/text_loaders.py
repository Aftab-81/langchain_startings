from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation",
    max_new_tokens = 1024
)

model = ChatHuggingFace(llm=llm)

template = PromptTemplate(
    template = "summarize the following {poem}",
    input_variables = ["poem"]
)



# Initialize TextLoader with raw string (r"...") to prevent escape character issues in Windows file paths
loader = TextLoader(r"Document_Loaders\sun.txt")

# TextLoader loads the entire text file into memory as a single Document object.
# Even though it is only 1 file, LangChain returns a List[Document] for standard interface consistency across all loaders.
#
# Internal structure of docs:
# docs = [
#     Document(
#         page_content="Golden sovereign of the day...", 
#         metadata={'source': 'Document_Loaders\\sun.txt'}
#     )
# ]

# Execute the synchronous load operation
docs = loader.load()

parser = StrOutputParser()

chain = template | model | parser

print("Summary:\n", chain.invoke({"poem": docs[0].page_content}))

# Note: This code might not work in this project environment because of some dependency issues. 
# But the code is correct and executable in order to run this create another project with
# only limited libraries installed. So that no dependency conflicts problem occur as it occurs
# in this environment.