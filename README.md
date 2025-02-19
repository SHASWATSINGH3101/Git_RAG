# RAG-Powered GitHub Code Ingestion and Query System

<img src="https://github.com/SHASWATSINGH3101/Git_RAG/blob/main/assets/e9601c1a-a2de-4c6f-8d4d-2ad7a609bdf0.jpg" alt="Guardian-Eye.AI" width="350">


This project provides a streamlined way to ingest GitHub repositories, process their contents, and use Retrieval-Augmented Generation (RAG) with a Language Model to answer questions about the code. This project runs locally and requires Ollama with the LLM of your choice installed.

## Features

- **GitHub Repository Ingestion**: Extracts code summaries, file structures, and content from a given GitHub repository.
- **File Storage**: Saves ingested data to a local file for persistence.
- **Retrieval-Augmented Generation (RAG)**: Uses a vector-based search system to query indexed documents.
- **LLM-Powered Queries**: Utilizes an Ollama-powered model to answer questions about the ingested codebase.
- **Web Interface**: A Flask-based front-end with Bootstrap styling for a clean UI.

## Installation

### Prerequisites

Ensure you have Python installed (recommended: Python 3.8+). Then install the required dependencies:

```bash
pip install flask llama-index torch gitingest
```

Additionally, install Ollama and the LLM of your choice:

```bash
ollama pull <your-llm-model>
```

### Running the Application

1. Clone the repository:
   ```bash
   git clone https://github.com/SHASWATSINGH3101/Git_RAG
   cd Git_RAG
   ```

2. Run the Flask application:
   ```bash
   python app.py
   ```

3. Open your browser and go to:
   ```
   http://127.0.0.1:5000/
   ```

## Usage

1. Enter a GitHub repository URL to ingest code.
2. Specify directories and embedding model parameters.
3. Input a query to retrieve insights about the code.
4. View results in the web interface.

## Recommended File Structure

```
project_root/
│── data/
│   ├── data.txt
│── index/
│── templates/
│   ├── index.html
│── static/
│   ├── styles.css
│── app.py
│── requirements.txt
│── README.md
```

## Configuration

You can modify key parameters in `app.py`:

- **Embedding Model**: Change `embed_model_name` to another supported model.
- **LLM Model**: Update `llm_model_name` for different LLM variants.
- **Index & Data Storage**: Adjust `index_dir` and `docs_dir` paths.

## Example Query

```
Which function is used for severity detection in the code?
```

## Future Improvements

- Implement user authentication
- Support for multiple repositories
- Additional visualization tools

## License

This project is licensed under the MIT License.

## Contributors

- My Github - [GitHub Profile](https://github.com/SHASWATSINGH3101)

