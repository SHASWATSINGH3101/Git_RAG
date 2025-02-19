import os.path
from typing import Optional, Tuple
from flask import Flask, request, render_template_string

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

app = Flask(__name__)

def process_with_gitingest(github_url: str) -> Tuple[str, str, str]:
    try:
        summary, tree, content = ingest(github_url)
        print(f'Ingested data from {github_url}')
        return summary, tree, content
    except Exception as e:
        print(f'Ingestion error: {e}')
        # Return default empty values so the app continues running
        return "", "", ""

def make_file_ingest(filepath: str, github_url: Optional[str] = None) -> None:
    if not github_url:
        print("No GitHub URL provided; ingestion skipped.")
        return

    summary, tree, content = process_with_gitingest(github_url)
    try:
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(summary + "\n")
            file.write(tree + "\n")
            file.write(content + "\n")
        print('Data written successfully.')
    except Exception as e:
        print("Error in writing data:", e)
        raise

def rag_setup(embed_model_name: str, llm_model_name: str):
    # Setup embedding model
    Settings.embed_model = resolve_embed_model(f"local:{embed_model_name}")
    # Setup LLM model
    Settings.llm = Ollama(model=llm_model_name, request_timeout=120.0)
    print('RAG setup complete.')

def get_index(index_dir: str, docs_dir: str):
    persist_dir = index_dir  # using lower-case variable name for clarity
    if not os.path.exists(persist_dir):
        documents = SimpleDirectoryReader(docs_dir).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=persist_dir)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
    return index

# Updated HTML template with Bootstrap styling
template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>RAG Query App</title>
    <!-- Bootstrap CSS CDN -->
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>
        body {
            margin-top: 30px;
            background-color: #f8f9fa;
        }
        .container {
            max-width: 800px;
        }
        .card {
            margin-top: 20px;
        }
        label {
            font-weight: 600;
        }
    </style>
</head>
<body>
<div class="container">
    <h1 class="text-center mb-4">RAG Query App</h1>
    <form method="post">
        <div class="form-group">
            <label for="github_url">GitHub URL for ingestion:</label>
            <input type="text" class="form-control" name="github_url" placeholder="Enter GitHub URL" value="{{ form_values.get('github_url','') }}">
        </div>
        <div class="form-group">
            <label for="data_path">Data file path:</label>
            <input type="text" class="form-control" name="data_path" placeholder="./data/data2.txt" value="{{ form_values.get('data_path','./data/data2.txt') }}">
        </div>
        <div class="form-group">
            <label for="index_dir">Index directory:</label>
            <input type="text" class="form-control" name="index_dir" placeholder="./index" value="{{ form_values.get('index_dir','./index') }}">
        </div>
        <div class="form-group">
            <label for="docs_dir">Docs directory:</label>
            <input type="text" class="form-control" name="docs_dir" placeholder="./data" value="{{ form_values.get('docs_dir','./data') }}">
        </div>
        <div class="form-group">
            <label for="embed_model_name">Embedding model name:</label>
            <input type="text" class="form-control" name="embed_model_name" placeholder="BAAI/bge-small-en-v1.5" value="{{ form_values.get('embed_model_name','BAAI/bge-small-en-v1.5') }}">
        </div>
        <div class="form-group">
            <label for="llm_model_name">LLM model name:</label>
            <input type="text" class="form-control" name="llm_model_name" placeholder="llama3.2:1b" value="{{ form_values.get('llm_model_name','llama3.2:1b') }}">
        </div>
        <div class="form-group">
            <label for="query_text">Query:</label>
            <input type="text" class="form-control" name="query_text" placeholder="Enter your query" value="{{ form_values.get('query_text','which function is used for severity detection in the code ?') }}">
        </div>
        <button type="submit" class="btn btn-primary btn-block">Run Query</button>
    </form>
    
    {% if result %}
    <div class="card">
        <div class="card-header">
            Query Response
        </div>
        <div class="card-body">
            <pre>{{ result }}</pre>
        </div>
    </div>
    {% endif %}
</div>
<!-- Bootstrap JS and dependencies -->
<script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.1/dist/umd/popper.min.js"></script>
<script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index_page():
    if request.method == "POST":
        # Retrieve form data
        github_url = request.form.get("github_url", "")
        data_path = request.form.get("data_path", "./data/data2.txt")
        index_dir = request.form.get("index_dir", "./index")
        docs_dir = request.form.get("docs_dir", "./data")
        embed_model_name = request.form.get("embed_model_name", "BAAI/bge-small-en-v1.5")
        llm_model_name = request.form.get("llm_model_name", "llama3.2:1b")
        query_text = request.form.get("query_text", "which function is used for severity detection in the code ?")
        
        # Ingestion step (only if a GitHub URL is provided)
        if github_url:
            make_file_ingest(data_path, github_url)
        else:
            print("No GitHub URL provided; skipping ingestion.")
        
        # RAG setup and query execution
        rag_setup(embed_model_name, llm_model_name)
        index_obj = get_index(index_dir, docs_dir)
        query_engine = index_obj.as_query_engine()
        response = query_engine.query(query_text)
        
        return render_template_string(template, result=response, form_values=request.form)
    else:
        # GET: render the form with default values
        return render_template_string(template, result=None, form_values={})

if __name__ == '__main__':
    app.run(debug=True)
