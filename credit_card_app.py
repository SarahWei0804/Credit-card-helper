import json 
from langchain_core.documents import Document
import re
from langchain.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder 
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
import time
import opencc
import ollama

ollama.pull("qwen2:7b-instruct")
ollama.pull("llama3.2")

bool_load_doc = False
bool_chunk = False


def load_docs(input_file: str):
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    documents = []
    for key in list(data.keys()):
        for d in data[key]:
            doc = Document(metadata={k:v for k,v in d.items() if k != 'page_content'}, page_content=re.sub(r'\t+', '\t', d['page_content']))
            documents.append(doc)
    print("Load documents successfully...")
    return documents

def chunk_and_store(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=300,
                                        length_function=len,
                                        is_separator_regex=True,
                                        chunk_overlap=50,
                                        separators=['。'])

    embedding = HuggingFaceEmbeddings(model_name='BAAI/bge-base-zh-v1.5', model_kwargs={'device':'mps'})
    collection_name = 'credit_card_chunk300'
    # qdrant_url = ':memory:'
    qdrant_url = 'http://localhost:6333'
    chunks = splitter.split_documents(documents)

    # vec_store = QdrantVectorStore.from_documents(chunks,
    #                             collection_name=collection_name,
    #                             embedding = embedding,
    #                             force_recreate=True,
    #                             location = qdrant_url)
    
    vec_store = QdrantVectorStore.from_existing_collection(
                              collection_name=collection_name,
                              embedding = embedding,
                              url = qdrant_url)
    
    print("Chunk and store in Qdrant successfully...")

    return vec_store

def get_retriever(enable_multiquery: bool, enable_selfquery: bool):
    global bool_load_doc
    global bool_chunk
    global vec_store
    global documents
    if bool_load_doc == False:
        documents = load_docs('documents.json')
        bool_load_doc = True
    if bool_chunk == False:
        vec_store = chunk_and_store(documents)
        bool_chunk = True
    metadata_field_info = [
        AttributeInfo(
            name="bank",
            description="The bank that hosts the credit card webpage",
            type="string"
        ),
        AttributeInfo(
            name="credit card",
            description="The credit card name the bank released",
            type="string"
        ),
        AttributeInfo(
            name="electronic payment",
            description="The electronic payment the bank offered discount",
            type="string"
        ),
    ]
    document_content_description = "The texts between HTML tags that is displayed on a credit card webpage, such as headings, paragraphs, links, and other textual elements"
    llm = ChatOllama(model='qwen2:7b-instruct', temperature=0, format='json')

    self_query_retriever = SelfQueryRetriever.from_llm(
        llm,
        vec_store,
        document_content_description,
        metadata_field_info,
        search_kwargs={'k': 10}
    )

    retriever = vec_store.as_retriever(search_kwargs={'k':5, 'score_threshold':0.5})
    
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=retriever, llm=llm
    )

    reranker = HuggingFaceCrossEncoder(model_name='BAAI/bge-reranker-base')
    compressor = CrossEncoderReranker(model=reranker, top_n=10)
    if enable_multiquery:
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever_from_llm)
    else:
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

    if enable_selfquery:
        ensemble_retriever = EnsembleRetriever(retrievers=[self_query_retriever, compression_retriever], weights=[0.6, 0.4])
        return ensemble_retriever
    else:
        return compression_retriever

def grade_documents(query, documents):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """
    ### Retrieval Grader
    # LLM
    llama = ChatOllama(model='llama3.2', temperature=0, format='json')

    prompt = PromptTemplate(
        template="""<<SYS>>You are a grader assessing relevance of a retrieved document to a user question.<</SYS>> \n 
        [INST]Here is the retrieved documents: \n\n {document} \n\n
        Here is the user question: {question} \n
        If the document contains keywords related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation. [/INST]""",
        input_variables=["question", "document"],
    )

    retrieval_grader = prompt | llama | JsonOutputParser()
    # Score each doc
    filtered_docs = []

    for d in documents:
        score = retrieval_grader.invoke(
            {"question": query, "document": d.page_content}
        )
        grade = score['score']
        if grade == 'yes':
            filtered_docs.append(d)
        else:
            pass
    print("Grade the retrieved documents successfully ...")
    return filtered_docs

def retrieve_docs(query: str, enable_grade: bool, enable_multiquery: bool, enable_selfquery: bool):
    retriever = get_retriever(enable_multiquery, enable_selfquery)
    retrieved_docs = retriever.invoke(query)
    if enable_grade:
        retrieved_docs = grade_documents(query, retrieved_docs)
        return retrieved_docs
    else:
        return retrieved_docs

def generate_reply(query):
    llm = ChatOllama(model='qwen2:7b-instruct', temperature=0)
    converter = opencc.OpenCC('s2t.json')
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI question-answering assistant. Your task is to answer the question based on the provided documents. The documents are part of the text from the description of the credit cards.
        The documents are not all relevant to the question. Please filter and reply with an answer. No pre-amble or explanation. Please output text only, not in markdown format.
        
        Documents: 
        {context}

        Question:
        {question}

        Answer:""",
    )

    def format_docs(docs):
        return "\n\n".join([f"DOCUMENT {index}\nTitle: {d.metadata['title']}\nReleased bank: {d.metadata['bank']}\n{d.page_content}" for index, d in enumerate(docs)])

    retrieved_docs = retrieve_docs(query, True, True, True)
    
    chain = (QUERY_PROMPT| llm| StrOutputParser())
    start = time.time()
    response = chain.invoke({"context": format_docs(retrieved_docs), "question":query})
    end = time.time()
    print(f"Spend time: {round(end-start, 2)}")
    return converter.convert(response)

if __name__ == "__main__":
    while True:
        query = input("Type your question about the credit cards..... ")
        if query != 'quit':
            print(generate_reply(query))

