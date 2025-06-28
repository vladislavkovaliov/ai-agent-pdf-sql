import os
import logging
import json

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langsmith import traceable
from tqdm import tqdm
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_URL = "http://192.168.1.141:11434"
PERSIST_DIRECTORY = "./data/chroma_db"


@traceable(run_type="llm", metadata={"ls_provider": "ollama", "model": "mistral"})
def create_qa_agent(pdf_path, model_name="mistral", persist_directory=PERSIST_DIRECTORY):
    persist_directory = persist_directory

    if os.path.exists(persist_directory):
        logging.info(f"Create a new database: {persist_directory}")

        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=OllamaEmbeddings(
                model=model_name,
                base_url=BASE_URL,
            ),
        )
    else:
        logging.info(f"Load already existing database: {persist_directory}")

        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        logging.info(f"Loaded {len(pages)} pages from the PDF.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        splits = text_splitter.split_documents(pages)

        logging.info(f"Split the document into {len(splits)} chunks.")

        embeddings = OllamaEmbeddings(
            model=model_name,
            base_url=BASE_URL,
        )
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )

        for i, chunk in enumerate(tqdm(splits, desc="Processing chunks"), 1):
            vectorstore.add_documents(
                documents=[chunk], # [chunk]
                embedding=embeddings,
            )

        logging.info(f"Stored {len(splits)} chunks in the vectorstore.")

    llm = OllamaLLM(
        model=model_name,
        base_url=BASE_URL,
    )

    prompt_template = """
    You are an expert assistant extracting structured data from pdf documents.

    Your task is to extract the following information in strict JSON format from the provided context.

    Schema:
    {{
      "categoryName": string,       // field name Услуга
      "amount": number,             // field name Сумма (only amount without currency)
      "createAt": string,           // field name Дата (in DATETIME YYYY-MM-DD HH-MM-SS format)
    }}

    Only extract the **Услуга**, **Дата**, **Сумма** 

    If you cannot find a field in the context, set it to null.

    Context:
    {context}

    Question:
    {question}

    Respond only in valid JSON using the schema above.
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={
                "k": 5,
            },
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    return qa_chain


@traceable(run_type="chain")
def ask_question(qa_chain, question):
    """
    Ask a question to the QA chain and get the response.

    Args:
        qa_chain: The QA chain created by create_qa_agent
        question (str): The question to ask

    Returns:
        dict: Response containing the answer and source documents
    """
    # try:
    #     response = qa_chain({"query": question})
    #     return {
    #         "answer": response["result"],
    #         "sources": [doc.page_content for doc in response["source_documents"]]
    #     }
    # except Exception as e:
    #     logging.error(f"An error occurred: {str(e)}")
    #     return {
    #         "error": f"An error occurred: {str(e)}",
    #         "answer": None,
    #         "sources": None
    #     }
    try:
        response = qa_chain.invoke({"query": question})

        raw_output = response["result"]

        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            parsed = {"error": "Invalid JSON", "raw_output": raw_output}

        return {
            "parsed": parsed,
            "sources": [doc.page_content for doc in response["source_documents"]]
        }
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

        return {
            "error": f"An error occurred: {str(e)}",
            "answer": None,
            "sources": None
        }


@traceable(run_type="llm", metadata={"ls_provider": "ollama", "model": "sqlcoder"})
def pre_processing_sql_query(payload):

    category_id_mapping = {
        "ЭЛЕКТРИЧЕСТВО": "21",
    }

    llm = OllamaLLM(
        model="mistral",
        base_url=BASE_URL,
    )

    prompt_template = PromptTemplate.from_template("""
    ### Instructions:
    
    ### Input:
    Extract values from JSON payload: categoryName, createAt, amount, generate insert SQL-script based on template.
    Match category name to category id based on mapping category name to category id. 
    In description insert categoryName and createAt.    
    This query will run on a database whose schema is represented in this string:
    
    SQL template:
    INSERT INTO payments (amount, description, currency, paymentMethod, categoryId, locationId) 
    VALUES (?, '', 'BYN', 'ONLINE', ?, 1);
    
    JSON:
    {payload}
    
    Mapping category name to category id:
    {category_id_mapping}

    ### Response:
    Based on your instructions, here is the SQL query:
    ```sql
    """)

    prompt = prompt_template.format(
        category_id_mapping=json.dumps(category_id_mapping, ensure_ascii=False),
        payload=json.dumps(payload, ensure_ascii=False),
    )

    sql_script = llm.invoke(prompt)

    print(f"SQL script:\n {sql_script}")


def main():
    # Ensure Ollama is running and the model is pulled
    # You can pull the model using: ollama pull mistral

    # Replace with your PDF path
    pdf_path = "./example.pdf"

    if not os.path.exists(pdf_path):
        logging.error(f"The file {pdf_path} does not exist.")

        return None

    # Create the QA agent
    qa_agent = create_qa_agent(pdf_path)

    # Ask questions from the user
    try:
        response = qa_agent.invoke(
            {"query": "Extract the patient's full name and the date of the medical event from this PDF."})

        raw_output = response["result"]

        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            parsed = {"error": "Invalid JSON", "raw_output": raw_output}

        # Prepare data to sql script
        pre_processing_sql_query(parsed)

        print("\n")
        print(f"Json data:\n {parsed}")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

        error = {
            "error": f"An error occurred: {str(e)}",
            "answer": None,
            "sources": None
        }

        print(error)


if __name__ == "__main__":
    main()