# Document Embeddings Generator Documentation

## Overview

`app_embeddings.py` is responsible for creating vector embeddings from PDF documents using LangChain and Google's Generative AI. It processes both course materials and syllabus documents, converting them into searchable vector databases that can be used by the question paper generator.

## Dependencies

- LangChain libraries for document processing and embeddings
- Google Generative AI for text embeddings
- FAISS for vector storage
- PyPDF Loader for PDF processing
- python-dotenv for environment management

## Key Components

```python
DATA_PATH_DATA = 'data/'
DATA_PATH_SYLLABUS = 'Syllabus/'
DB_FAISS_PATH_DATA = 'vectorstore_data/db_faiss'
DB_FAISS_PATH_SYLLABUS = 'vectorstore_Syllabus/db_faiss'
```

- Defines paths for input PDF documents
- Specifies locations for storing vector databases

### 2. Environment Setup

```python
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('API_KEY')
genai.configure(api_key=os.getenv("API_KEY"))
```

- Loads environment variables
- Configures Google API credentials

### 3. Main Function: `create_vector_db(Path, FAISS_PATH)`

#### Document Loading

```python
loader = DirectoryLoader(Path, glob='.pdf', loader_cls=PyPDFLoader)
documents = loader.load()
```

- Loads all PDF files from specified directory
- Uses PyPDFLoader for PDF processing

#### Text Processing

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
texts = text_splitter.split_documents(documents)
```

- Splits documents into manageable chunks
- Uses 512 characters per chunk with 64 character overlap
- Ensures context preservation between chunks

#### Embedding Generation

```python
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
```

- Initializes Google's text embedding model
- Converts text chunks into vector embeddings

#### Batch Processing

- Processes documents in batches of 50 chunks
- Features:
  - Incremental processing
  - Error handling
  - Progress tracking
  - Memory efficient

#### Vector Store Creation

- Creates FAISS vector database
- Adds embeddings incrementally
- Saves processed embeddings to disk

## Process Flow

1. **Input**: PDF documents from specified directories
2. **Processing**:
   - Load PDFs
   - Split into text chunks
   - Generate embeddings
   - Create vector store
3. **Output**: FAISS vector database files

## Usage

Run the script directly to process both course materials and syllabus:

```python
if name == "main":
create_vector_db(DATA_PATH_DATA, DB_FAISS_PATH_DATA)
create_vector_db(DATA_PATH_SYLLABUS, DB_FAISS_PATH_SYLLABUS)
```

## Error Handling

- Validates text chunk format
- Handles batch processing errors
- Provides detailed error messages

## Performance Considerations

- Batch processing to manage memory usage
- Error recovery capabilities
- Progress tracking for long operations

## Output

Creates two FAISS vector databases:

1. Course material embeddings (`vectorstore_data/db_faiss`)
2. Syllabus embeddings (`vectorstore_Syllabus/db_faiss`)

## Important Notes

- Requires proper setup of Google API credentials
- Sufficient disk space for vector databases
- PDF files must be readable and properly formatted
- Internet connection required for embedding generation
