import json 
from langchain_core.documents import Document
import re
from langchain_ollama import ChatOllama
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
import gradio as gr
# import opencc
import ollama
import os
from qdrant_client import QdrantClient
from langchain_qdrant import FastEmbedSparse, RetrievalMode
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain.tools.retriever import create_retriever_tool


import sys
sys.path.append("utility")
from agentic_rag import AIAgent
# ollama.pull("qwen2.5:7b-instruct")
# ollama.pull("llama3.2")
llm = ChatOllama(model='qwen2.5:7b-instruct', temperature=0, format='json')
bool_load_doc = False
bool_chunk = False

class Processor:
    def __init__(self):
        pass
    def load_docs(self, input_file: str)->List[Document]:
        with open(input_file, 'r', encoding='utf-8') as file:
            data = json.load(file)

        documents = []
        for key in list(data.keys()):
            for d in data[key]:
                doc = Document(metadata={k:v for k,v in d.items() if k != 'page_content'}, page_content=re.sub(r'\t+', '\t', d['page_content']))
                documents.append(doc)
        print("Load documents successfully...")
        return documents

    def chunk_and_store(self, documents:List[Document]):
        global embedding
        print("Start chunking....")
        splitter = RecursiveCharacterTextSplitter(chunk_size=300,
                                            length_function=len,
                                            is_separator_regex=True,
                                            chunk_overlap=50,
                                            separators=['。'])
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        # BAAI/bge-base-zh-v1.5
        embedding = HuggingFaceEmbeddings(model_name='BAAI/bge-m3', model_kwargs={'device':'cpu'})
        collection_name = 'credit_card_chunk300_hybrid_m3'
        qdrant_url = 'http://localhost:6333'
        chunks = splitter.split_documents(documents)
        qdrant_client = QdrantClient(url=qdrant_url)
        if qdrant_client.collection_exists(collection_name=collection_name):
            vec_store = QdrantVectorStore.from_existing_collection(collection_name=collection_name,
                                                            embedding=embedding,
                                                            url=qdrant_url)
        else:
            ## check if the index exists already
            vec_store = QdrantVectorStore.from_documents(chunks,
                                        collection_name=collection_name,
                                        embedding = embedding,
                                        force_recreate=True,
                                        url = qdrant_url,
                                        sparse_embedding=sparse_embeddings,
                                        retrieval_mode=RetrievalMode.HYBRID
                                        )

        
        print("Chunk and store in Qdrant successfully...")

        return vec_store

    def get_retriever(self, enable_multiquery: bool, enable_selfquery: bool, enable_parentdoc: bool):
        global bool_load_doc
        global bool_chunk
        global vec_store
        global documents
        if bool_load_doc == False:
            documents = self.load_docs('documents.json')
            bool_load_doc = True
        if bool_chunk == False:
            vec_store = self.chunk_and_store(documents)
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

        print("Get retriever...")
        if enable_parentdoc:
            store = InMemoryStore()
            child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
            #
            #create bigger chunks
            parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
            # embedding = HuggingFaceEmbeddings(model_name='BAAI/bge-m3', model_kwargs={'device':'mps'})
            vectordb = Chroma(
                    embedding_function=embedding,
                    persist_directory="./chroma_db_parent",
                    collection_name="split_parents_m3",
                )

            big_chunks_retriever = ParentDocumentRetriever(
                vectorstore=vectordb,
                docstore=store,
                child_splitter=child_text_splitter,
                parent_splitter=parent_splitter,
            )
            big_chunks_retriever.add_documents(documents)

            return big_chunks_retriever
        
        if enable_multiquery and enable_selfquery:
            compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever_from_llm)
            ensemble_retriever = EnsembleRetriever(retrievers=[self_query_retriever, compression_retriever], weights=[0.6, 0.4])
            return ensemble_retriever
        elif enable_multiquery:
            compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever_from_llm)
        elif enable_selfquery:
            compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=self_query_retriever)
        else:
            compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
        return compression_retriever



    
if __name__ == "__main__":
    enable_multiquery = True
    enable_selfquery = True
    enable_parentdoc = False
    processor = Processor()
    retriever = processor.get_retriever(enable_multiquery, enable_selfquery, enable_parentdoc)

    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_credit_cards",
        "Search and return information about credit cards from 國泰銀行, 星展銀行, 永豐銀行跟第一銀行.",
    )


    tools = [retriever_tool]

    agent = AIAgent(tools)
    graph = agent.define_flow()
    
    # while True:
    #     query = input("Type your question about the credit cards....  ")
    #     if query != 'quit':
            
            # print(generate_reply(query))

    def echo(message, history):
        output = graph.invoke({"messages":[("user", message),], "tools":tools})
        return output['messages'][-1].content

    demo = gr.ChatInterface(fn=echo, type="messages", examples=["星展eco永續極簡卡優惠有什麼？", 
                                                                "星展PChome Prime聯名卡回饋是幾趴？", 
                                                                "星展eco永續優選卡活動期間是什麼時候？"], title="Credit Card Bot")
    demo.launch(server_name='0.0.0.0', server_port=8888)