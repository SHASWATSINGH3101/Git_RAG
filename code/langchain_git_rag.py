from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph

import os.path
from typing import Optional, Tuple, List, TypedDict

import chromadb
from gitingest import ingest

# Global variable for vector store
vector_store = None


def process_with_gitingest(github_url: str) -> Tuple[str, str, str]:
    try:
        summary, tree, content = ingest(github_url)
        print(f'ingested data from {github_url}')
        return summary, tree, content
    except Exception as e:
        print('Ingestion error')
    
def make_file_ingest(filepath: str, github_url: Optional[str] = None) -> None:
    if not github_url:
        print('No url, ingestion Skipped')
        return
    summary, tree, content = process_with_gitingest(github_url)
    try:
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(summary)
            file.write(tree)
            file.write(content)
        print('data written')
    except Exception as e:
        print('error in writing data')
        raise

def rag_setup(embed_model_name, llm_model_name, choice):
    # Embedding model
    embeddings = HuggingFaceEmbeddings(model_name=embed_model_name)
    # LLM model
    if choice == 'online':
        llm = ChatGroq(
            model=llm_model_name,
            temperature=0.7,
            max_tokens=265,
            api_key='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'
        )
    elif choice == 'offline':
        llm = ChatOllama(
            model='llama3-groq-tool-use:latest',
            temperature=0.7,
            num_predict=256
        )
    return llm, embeddings

print('RAG setup complete')

def get_index(index_dir, data_path, embeddings):
    persist_dir = index_dir
    # If the persist directory does NOT exist, create it and build a new index
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir, exist_ok=True)
        loader = DirectoryLoader(
            data_path,
            glob='**/*.txt',
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        all_splits = text_splitter.split_documents(docs)
        vector_store = Chroma(
            collection_name='example_collection',
            embedding_function=embeddings,
            persist_directory=persist_dir
        )
        _ = vector_store.add_documents(documents=all_splits)
        print("New Indexes made!")
    else:
        # If the persist directory exists, load the existing index
        client = chromadb.PersistentClient(path=persist_dir)
        # collection = client.get_collection('example_collection')
        vector_store = Chroma(
            client=client,
            collection_name='example_collection',
            embedding_function=embeddings
        )
        print('Existing indexes used!')
    return vector_store




template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""

custom_rag_prompt = PromptTemplate.from_template(template)

class State(TypedDict):
    questions: str
    context: str
    answer: str
    llm: object  # storing llm for generation

def retrieve(state: State):
    # Use the global vector_store
    retrieved_docs = vector_store.similarity_search(state['questions'])
    return {"context": retrieved_docs}

def generate(state: State):
    # 'context' holds the list of retrieved documents
    llm = state['llm']
    docs_content = '\n\n'.join(doc.page_content for doc in state['context'])
    messages = custom_rag_prompt.invoke({
        'question': state['questions'],
        'context': docs_content
    })
    response = llm.invoke(messages)
    return {'answer': response}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, 'retrieve')
graph = graph_builder.compile()

def query_engine(query: str, llm):
    initial_state = {
        "questions": query,
        "context": "",
        "answer": "",
        "llm": llm
    }
    result = graph.invoke(initial_state)
    # Return only the answer text
    if hasattr(result['answer'], 'content'):
        return result['answer'].content
    else:
        return result['answer']


def main() -> None:
    data_path = r'X:\\VS_CODE\\_SIDE Projects\\LinkdIn post BOT\\codebase\\IPYNB\data'
    github_url = ""
    index_dir = r'X:\\VS_CODE\\_SIDE Projects\\LinkdIn post BOT\\codebase\\IPYNB\\chromadb'
    docs_dir = r"X:\\VS_CODE\\_SIDE Projects\\LinkdIn post BOT\\codebase\\GIT_rag\\data"
    
    embed_model_name = 'sentence-transformers/all-mpnet-base-v2'
    llm_model_name = 'llama3-70b-8192'
    choice = 'online'
    query_text = 'which messaging service was used for emergency reporting in this project ?'

    make_file_ingest(data_path, github_url if github_url else None)
    llm, embeddings = rag_setup(embed_model_name, llm_model_name, choice)
   
    global vector_store
    vector_store = get_index(index_dir, data_path, embeddings)
    
    response = query_engine(query_text, llm)
    print(response)
    
if __name__ == '__main__':
    main()
