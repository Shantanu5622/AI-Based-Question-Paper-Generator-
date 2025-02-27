from langchain.prompts import PromptTemplate
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnablePassthrough
import os
from datetime import datetime

# Setting Google API Key
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('API_KEY')
genai.configure(api_key=os.getenv("API_KEY"))

# Path of vectore database
DB_FAISS_PATH_DATA = 'vectorstore_data/db_faiss'
DB_FAISS_PATH_QP = 'vectorstore_Syllabus/db_faiss'

# Set up Google LLM
def load_llm():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.6)
    return llm

# Create a custom prompt template
question_paper_template = """
You are an experienced professor tasked with creating a question paper for Software Engineering. Follow these instructions precisely:

1. Use the provided context and syllabus to generate questions.

2. Each question MUST be based on specific units as follows:
   - Q.1 (MCQs): From UNIT 1 to 5 
   - Q.2: From UNIT 1 only
   - Q.3: From UNIT 2 only
   - Q.4: From UNIT 3 only
   - Q.5: From UNIT 4 only
   - Q.6: From UNIT 5 only

3. Question Format Requirements:
   - Q.1: Generate 12 multiple choice questions
     * Each MCQ must have 4 options (a, b, c, d)
     * Each MCQ should end with (1)
     * Each MCQ must be seperated by 1 new line
   
   - Q.2 to Q.6: 
     * For Q.2 and Q.3: Two sub-questions (A and B)
     * Each sub-question should end with (6)
     * For Q.4, Q.5, and Q.6: Three sub-questions (A, B, and C)
     * Each sub-question should end with (6)
     * Students attempt any two from A, B, C
     * Students attempt any two from A, B


4. Question Format Example:
   Q.2 Solve the following:
   A) Explain the concept of requirements engineering and its importance in software development. (6)
   B) Discuss various requirements elicitation techniques with examples. (6)

   Q.4 Solve any TWO of the following:
   A) What is system modeling? Explain its significance in software development. (6)
   B) Describe the different types of UML diagrams with their purposes. (6)
   C) Explain the concept of behavioral modeling with suitable examples. (6)

5. Question Guidelines:
   - Questions should test different cognitive levels (understanding, application, analysis)
   - Sub-questions should be related but test different aspects of the same unit
   - Questions should be clear, unambiguous, and appropriate for a 3-hour examination
   - Include a mix of theoretical and practical questions where applicable

6. General Instructions:
   - Format the entire question paper using Markdown syntax.
   - Ensure each MCQ option and sub-question is on a new line.
   - Add a blank line (normal new line) between sub-questions for proper spacing.
   - Use proper indentation for MCQ options (4 spaces for each option).
   - Add horizontal lines (---) to separate sections like Instructions, Questions, and Answers for clarity.

Use this structure for the question paper:

<div align="center">

DR. BABASAHEB AMBEDKAR TECHNOLOGICAL UNIVERSITY, LONERE  
Regular/Supplementary Winter Examination – 2024  

Course: Computer Engineering  
Subject Code & Name: BTCOC501: Software Engineering  
Branch: Computer Engineering  
Semester: V  

Time: 3 Hours                                                                     Max. Marks: 60  

</div>

---

### Instructions:  
1. All questions are compulsory.  
2. Figures to the right indicate full marks.  
3. Assume suitable data if necessary.  

---


[Rest of the header and instructions as provided]

Context for question generation:
{context}

Syllabus structure:
{qp_structure}

Generate a complete question paper following all the above requirements strictly, ensuring each question/sub-question ends with marks in parentheses.
"""

def generate_question_paper(book_data, qp_structure):
    prompt = PromptTemplate(template=question_paper_template, input_variables=['context', 'qp_structure'])
    llm = load_llm()
    
    # Create the chain using the newer approach
    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
        document_variable_name="context"
    )
    
    # Create a Document object from the context
    context = book_data + "\n\n" + qp_structure
    doc = Document(page_content=context)
    
    # Create a runnable pipeline
    chain = (
        RunnablePassthrough.assign(context=lambda x: x["input_documents"])
        | document_chain
    )
    
    # Invoke the chain
    response = chain.invoke({
        "input_documents": [doc],
        "qp_structure": qp_structure
    })
    
    return response


# Load the embeddings and data
def load_embeddings():
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004")
    book_db = FAISS.load_local(DB_FAISS_PATH_DATA, embeddings, allow_dangerous_deserialization=True)
    qp_db = FAISS.load_local(DB_FAISS_PATH_QP, embeddings, allow_dangerous_deserialization=True)

    # Load data from FAISS
    book_data = book_db.similarity_search("extract relevant book data")[0].page_content
    qp_structure = qp_db.similarity_search("extract relevant question paper structure")[0].page_content
    return book_data, qp_structure




def gen_qp():
    # Define folder to save the question paper
    output_folder = "generated_question_papers"
    
    # Create the folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"question_paper_{timestamp}.md"
    file_path = os.path.join(output_folder, file_name)
    
    # Load embeddings
    print("Loading embeddings...")
    try:
        book_data, qp_structure = load_embeddings()
    except Exception as e:
        print(f"Error during embeddings loading: {e}")
        return
    
    # Generate question paper
    print("Generating question paper...")
    try:
        question_paper = generate_question_paper(book_data, qp_structure)
    except Exception as e:
        print(f"Error during question paper generation: {e}")
        return
    
    # Save the question paper
    print("Saving question paper...")
    try:
        with open(file_path, "w", encoding='utf-8') as file:
            file.write(question_paper)
        print(f"Question paper generated and saved successfully at {file_path}!")
    except Exception as e:
        print(f"Failed to save question paper: {e}")

gen_qp()        

# choice = input('Enter 1 for generating question paper and 2 for evaluating answer key: ')
# if choice == '1':
#     gen_qp()
# else:
#     print('Choice not valid')
