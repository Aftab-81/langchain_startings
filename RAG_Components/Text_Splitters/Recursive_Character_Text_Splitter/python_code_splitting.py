from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

splitter = RecursiveCharacterTextSplitter.from_language(
    language = Language.PYTHON,
    chunk_size = 70,
    chunk_overlap = 10
)

code = """
    class Student:

    @classmethod
    def show(cls):
        print(type(cls))

    Student.show()
    print(type(Student))
"""

chunks = splitter.split_text(text = code)

print(len(chunks))
print(chunks)