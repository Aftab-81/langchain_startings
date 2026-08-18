from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separators = ""
)

# Less no. of chunk size there are more numbers of chunks 

text = """
    Text splitting is the process of breaking a long document into smaller, 
    easier-to-handle parts. Instead of giving the entire document to an 
    AI system all at once — which might be too much to process — text splitting helps divide 
    the content into chunks of a manageable size.

    These chunks are usually based on sentences, paragraphs, or character limits 
    and sometimes include some overlap so the system doesn’t lose the meaning that 
    flows from one part to the next.

"""

chunks = splitter.split_text(text = text)

print(len(chunks))
print(chunks)