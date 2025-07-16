import os
import logging
import json
import re
import subprocess
import sys
import markdown


from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
# from langchain
from weasyprint import HTML
from langsmith import traceable
from datetime import datetime, date
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader, Template
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')

# BASE_URL = "http://192.168.1.141:11434"
BASE_URL = "http://192.168.1.70:11434"
PERSIST_DIRECTORY = "./data/chroma_db"

sql_command_list = []

PATH_HTML_FILE = "./output.html"

def write_html_to_file(html, path = PATH_HTML_FILE):
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

def render_html(sql_scripts):
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("template.html")
    scripts = [item["sql"] for item in sql_scripts]

    rendered_html = template.render(
        title="Мой сайт",
        heading="Список задач",
        sql_scripts=scripts
    )

    return rendered_html

def convert_html_to_pdf(html_content, path = f"./reports/report_{datetime.now().strftime("%d.%m.%Y_%H-%M-%S")}.pdf"):
    HTML(string=html_content).write_pdf(path)


def clean_json(json_str):
    json_str = re.sub(r'//.*', '', json_str)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)

    return json_str


@traceable(run_type="llm", metadata={"ls_provider": "ollama", "model": "mistral"})
def create_qa_agent(pdf_path, model_name="mistral", persist_directory=PERSIST_DIRECTORY):
    persist_directory = persist_directory + "_" + pdf_path.split("/")[-1].split(".")[0]

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
                documents=[chunk],
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
      "categoryName": string,       // поле "Услуга"
      "amount": number,             // поле "Сумма" (only amount without currency)
      "createAt": string,           // поле "Дата" (in DATETIME YYYY-MM-DD HH-MM-SS format)
    }}

    Only extract the **Услуга**, **Дата**, **Сумма** 

    If you cannot find a field in the context, set it to null.

    Context:
    {context}

    Question:
    {question}

    Respond only in valid JSON using the schema above. 
    The JSON **must not contain any comments, explanations, or annotations**.
    Only output raw JSON. Do not add any `//` or `/* */` style comments.
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


@traceable(run_type="llm", metadata={"ls_provider": "ollama", "model": "mistral"})
def pre_processing_sql_query(payload):
    category_id_mapping = {
        "КОММУНАЛЬНЫЕ-ПЛАТЕЖИ": 24,
        "КОММУНАЛКА": 22,
        "Дополнительные платежи": 25,
        "Электроэнергия (физ.лица)": 21,
        "Интернет, ТВ": 27,
        "МТС": 28,
        "Центральный парк-Паркинг-2": 30
    }

    llm = OllamaLLM(
        model="mistral",
        base_url=BASE_URL,
    )

    prompt_template = PromptTemplate.from_template("""
    ### Instructions:

    You are a SQL generator. Extract values from JSON payload: categoryName, createAt, amount.
    
    Then generate an SQL INSERT statement using the provided template.
    
    **IMPORTANT:**
    - Generate an SQL INSERT statement using the provided template.
    - Do not return extra commentary — only a valid SQL query inside a code block.
    - The *description* value should be filled by categoryName - createAt
    - 
    
    ### SQL Template:
    INSERT INTO payments (amount, description, currency, paymentMethod, categoryId, locationId)
    VALUES (?, '', 'BYN', 'ONLINE', ?, 1);
    
    ### JSON Input:
    {payload}
    
    ### Category Mapping:
    {category_id_mapping}
    
    ### Response (Strict) only with a SQL query:
    <SQL_QUERY_HERE>
    """)

    prompt = prompt_template.format(
        category_id_mapping=json.dumps(category_id_mapping, ensure_ascii=False),
        payload=json.dumps(payload, ensure_ascii=False),
    )

    sql_script = llm.invoke(prompt)

    return sql_script


def main(path):
    if not os.path.exists(path):
        logging.error(f"The file {path} does not exist.")

        return None

    # Create the QA agent
    qa_agent = create_qa_agent(path)

    # Ask questions from the user
    try:
        response = qa_agent.invoke({
            "query": "Extract the values for service name (category), amount (number only), and date (in YYYY-MM-DD HH:MM:SS format) from this document"
        })

        raw_output = response["result"]

        try:
            cleaned_output = clean_json(raw_output)
            parsed = json.loads(cleaned_output)
        except json.JSONDecodeError:
            parsed = {"error": "Invalid JSON", "raw_output": raw_output}

        # Prepare data to sql script
        sql_script = pre_processing_sql_query(parsed)

        print(f"SQL script:\n {sql_script}")
        print(f"Json data:\n {parsed}\n")

        sql_command_list.append({
            "sql": sql_script,
            "json": parsed,
            "path": path,
        })


    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

        error = {
            "error": f"An error occurred: {str(e)}",
            "answer": None,
            "sources": None
        }

        print(error)


if __name__ == "__main__":
    file_path = sys.argv[1]

    print(f"Processing files {sys.argv[1:]}")

    for path in sys.argv[1:]:
        main(path)

    render_html = render_html(sql_scripts=sql_command_list)

    write_html_to_file(render_html)

    convert_html_to_pdf(render_html)