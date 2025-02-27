# Question Paper Generator Documentation

## Overview

`qp_gen.py` is a Python script that automates the generation of question papers using LangChain and Google's Generative AI. The system uses vector databases to store and retrieve course material and syllabus information, then generates structured question papers based on predefined templates.

## Dependencies

- LangChain libraries for AI operations
- Google Generative AI (Gemini model)
- FAISS for vector storage
- PyMuPDF (fitz) for PDF operations
- python-dotenv for environment management

## Key Components

### 1. File Path Configuration

```python
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('API_KEY')
genai.configure(api_key=os.getenv("API_KEY"))
```

- Loads environment variables
- Configures Google API credentials
- Sets up paths for vector databases

### 2. LLM Configuration

```python
def load_llm():
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.6)
return llm
```

- Initializes Google's Gemini model
- Sets temperature to 0.6 for balanced creativity and consistency

### 3. Question Paper Template

The `question_paper_template` is a comprehensive prompt that includes:

- Strict instructions for question generation
- Unit-wise question distribution
- Formatting requirements
- Marking scheme
- Example question formats
- Guidelines for question quality

### 4. Core Functions

#### `generate_question_paper(book_data, qp_structure)`

- Creates a prompt template
- Initializes the LLM
- Sets up document chain for processing
- Processes input data and generates question paper
- Returns formatted response

```python
def generate_question_paper(book_data, qp_structure):
# Creates document chain and processes input
# Returns generated question paper
```

#### `load_embeddings()`

- Initializes Google's text embedding model
- Loads vector databases for book data and question paper structure
- Retrieves relevant content using similarity search
- Returns processed book data and question paper structure

```python
def load_embeddings():
# Loads and processes embeddings
# Returns book_data and qp_structure
```

#### `gen_qp()`

- Main function that orchestrates the question paper generation
- Calls load_embeddings() to get data
- Generates question paper using generate_question_paper()
- Saves output to a text file

```python
def gen_qp():
# Orchestrates the question paper generation process
# Saves output to file
```

## Data Storage

The script uses two FAISS vector databases:

- `DB_FAISS_PATH_DATA`: Stores course material
- `DB_FAISS_PATH_QP`: Stores question paper structure/syllabus

## Output

- Generates a structured question paper following university format
- Saves output to 'question_paper.txt'
- Includes:
  - University details
  - Course information
  - Instructions
  - Questions with proper marking scheme
  - Unit-wise distribution of questions

## Note

- Ensure proper setup of environment variables
- Vector databases must be properly initialized with course material
- Internet connection required for API calls to Google's services
