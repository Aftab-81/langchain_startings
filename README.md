# LangChain Models & Output Demonstrations

This repository contains basic demonstrations of **LLMs, Chat Models, Embedding Models, and Structured/Unstructured Outputs** using Python and LangChain.

The purpose of this project is to understand the fundamental model components used when building **LLM and Generative AI applications** with LangChain.

## 📚 Topics Covered

### 1. LLMs

Demonstration of using a traditional **Large Language Model (LLM)** through LangChain.

The demo covers:

* Connecting an LLM with LangChain
* Sending a prompt to the model
* Generating text-based responses
* Understanding the basic LLM workflow

**Basic workflow:**

```text
Prompt → LLM → Text Response
```

---

### 2. Chat Models

Demonstration of **Chat Models**, which are designed to work with conversational messages.

The demo covers:

* Creating a Chat Model
* Using system, human, and AI messages
* Sending messages to the model
* Receiving model responses

**Basic workflow:**

```text
Messages → Chat Model → AI Response
```

Chat Models are commonly used for:

* Chatbots
* AI assistants
* Question-answering systems
* Conversational applications

---

### 3. Embedding Models

Demonstration of **Embedding Models**, which convert text into numerical vector representations.

For example:

```text
"Machine Learning"
        ↓
Embedding Model
        ↓
[0.12, -0.43, 0.78, ...]
```

The demo helps understand how text can be converted into vectors that capture semantic meaning.

Embeddings are commonly used in:

* Semantic search
* Vector databases
* Retrieval-Augmented Generation (RAG)
* Document similarity
* Recommendation systems

---

### 4. Unstructured Output

The project demonstrates how an LLM can generate **unstructured text output**.

For example:

```text
Machine Learning is a subset of Artificial Intelligence
that enables computers to learn patterns from data...
```

There is no predefined schema that the response must follow.

**Workflow:**

```text
Prompt → LLM → Free-form Text
```

Unstructured output is useful when the application needs natural language responses such as:

* Explanations
* Summaries
* Essays
* General conversations

---

### 5. Structured Output

The project also demonstrates **structured output using Pydantic**.

A Pydantic model can define the expected structure of the LLM response.

Example:

```python
from pydantic import BaseModel

class Student(BaseModel):
    name: str
    age: int
    course: str
```

Instead of receiving free-form text, the application expects data following this structure:

```text
Student(
    name="Aftab",
    age=21,
    course="Computer Science"
)
```

This makes the LLM response easier for a Python application to:

* Validate
* Process
* Store in a database
* Pass to another component
* Use programmatically

**Workflow:**

```text
Prompt
   ↓
LLM
   ↓
Structured Output
   ↓
Pydantic Object
```

> **Note:** The structured-output demonstration also explores the limitations of structured output support with different LangChain model integrations.

---

## 🗂️ Project Structure

```text
langchain_models/
│
├── LLMs/
│   └── ...
│
├── ChatModels/
│   └── ...
│
├── EmbeddingModels/
│   └── ...
│
├── PydanticDemo/
│   └── pydantic_demo.py
│
├── .env
├── .gitignore
└── README.md
```

> The exact filenames may vary depending on the demos added to the project.

---

## 🛠️ Technologies Used

* **Python**
* **LangChain**
* **LangChain Hugging Face**
* **Pydantic**
* **Hugging Face**
* **python-dotenv**

---

## 🔐 Environment Variables

API keys are loaded using `python-dotenv`.

Create a `.env` file in the project root:

```env
HUGGINGFACEHUB_API_TOKEN=your_api_key_here
```

Do **not** commit your `.env` file to GitHub.

Add it to `.gitignore`:

```gitignore
.env
venv/
__pycache__/
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd langchain_models
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🎯 Learning Objective

The main objective of this project is to understand the fundamental **model components in LangChain** before moving toward more advanced Generative AI concepts.

The concepts demonstrated here form the foundation for technologies such as:

```text
LLMs
  ↓
Chat Models
  ↓
Embeddings
  ↓
Structured Outputs
  ↓
Retrievers
  ↓
Vector Databases
  ↓
RAG
  ↓
Agents
  ↓
Generative AI Applications
```

---

## 📌 Summary

| Concept                 | Purpose                                      |
| ----------------------- | -------------------------------------------- |
| **LLM**                 | Generates text from prompts                  |
| **Chat Model**          | Handles conversational messages              |
| **Embedding Model**     | Converts text into vector representations    |
| **Unstructured Output** | Generates free-form text                     |
| **Structured Output**   | Generates data following a predefined schema |
| **Pydantic**            | Defines and validates structured data        |

---

## 👨‍💻 Author

**Aftabalam Makandar**

This project is part of my learning journey in **Generative AI, LLMs, NLP, and LangChain**.
