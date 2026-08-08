from langchain_core.prompts import PromptTemplate

template = PromptTemplate(

            template = 
                """
                    Please summarize the research paper titled "{paper_input}" with the following specifications:
                    
                    Explanation Style: {style_input}
                    Explanation Length: {length_input}
                    
                    1. Mathematical Details:
                    - Include relevant mathematical equations if present in the paper.
                    - Explain mathematical concepts using simple, intuitive explanations where applicable.
                    
                    2. Analogies:
                    - Use relevant analogies to simplify complex ideas.
                    
                    3. Key Information:
                    - Identify and explain the main objectives, methodology, results, and conclusions.
                    - Focus on the most important information according to the specified explanation style and length.
                    
                    4. Accuracy:
                    - Do not guess or invent information that is not available in the paper.
                    - If certain information is not available in the paper, respond with: "Insufficient information available."
                    
                    Ensure the summary is clear, accurate, well-structured, and aligned with the provided style and length.
                """,
                input_variables = ["paper_input", "style_input", "length_input", "math_input", "abcd"],
                validate_template = True
            )

template.save("Langchain_PROMPTS/prompt_template.json")
print("Prompt template is saved successfully")