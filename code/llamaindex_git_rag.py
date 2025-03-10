import os.path
from typing import Optional, Tuple

from gitingest import ingest
from llama_index.core import (

    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings, 
)

from llama_index.core.embeddings import resolve_embed_model
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama


def process_with_gitingest(github_url: str) -> Tuple[str,str,str]:
    try:
        summary, tree, content = ingest(github_url)
        print(f'ingested data from{github_url}')
        return summary, tree, content
    except Exception as e:
        print(f'Ingestion error')

def make_file_ingest(filepath:str, github_url: Optional[str] = None)-> None:
    
    if not github_url:
        print("No url,ingestion Skipped")
        return
    
    summary, tree, content = process_with_gitingest(github_url)
    try:
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(summary)
            file.write(tree)
            file.write(content)
        print('Data written')
    except Exception as e:
        print("error inwriting data")
        raise


## RAG

def rag_setup(embed_model_name, llm_model_name):
    # Embedding model
    Settings.embed_model = resolve_embed_model("local:{embed_model_name}".format(embed_model_name=embed_model_name))
    #  llm model
    Settings.llm = Ollama(model=llm_model_name, request_timeout=120.0)
    print('Rag setup complete')

def get_index(index_dir, docs_dir):
    Persist_dir = index_dir
    if not os.path.exists(Persist_dir):
        documents = SimpleDirectoryReader(docs_dir).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=Persist_dir)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=Persist_dir)
        index = load_index_from_storage(storage_context)
    return index

def main() -> None:

    data_path = 'X:\VS_CODE\_SIDE Projects\LinkdIn post BOT\data\data2.txt'
    github_url = ""
    index_dir = 'X:\VS_CODE\_SIDE Projects\LinkdIn post BOT\codebase\GIT_rag\index'
    docs_dir = "X:\VS_CODE\_SIDE Projects\LinkdIn post BOT\codebase\GIT_rag\data"
    embed_model_name = 'BAAI/bge-small-en-v1.5'
    llm_model_name = 'phi3:3.8b'
    query_text = 'which type of model was used for violence detection in this project  ?'


    make_file_ingest(data_path, github_url if github_url else None)
    rag_setup(embed_model_name, llm_model_name)
    index = get_index(index_dir, docs_dir)

    query_engine = index.as_query_engine()
    response = query_engine.query(query_text)
    print("Query response:")
    print(response)

if __name__ == '__main__':
    main()