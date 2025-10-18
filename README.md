# Chat with My Research 📄🤖

This project is a web application that allows users to have a conversation with my NLP research paper, "Paraphrase Detection in a Low Resourced Language: Kannada." It uses a Retrieval-Augmented Generation (RAG) pipeline to provide accurate, context-aware answers based solely on the content of the document, effectively preventing the language model from hallucinating.

This application was built to showcase an end-to-end MLOps workflow, from data processing and model integration to building an interactive user interface.

## Features

-   **Interactive Chat Interface:** Ask questions in natural language and receive answers from the research paper.
-   **Retrieval-Augmented Generation (RAG):** The system retrieves relevant passages from the paper before generating an answer, ensuring all responses are grounded in the source text.
-   **Dynamic PDF Processing:** Loads and processes the source PDF, creating a searchable vector database for efficient retrieval.
-   **Secure API Key Management:** Uses Streamlit's secrets management to keep API keys safe.

## Tech Stack

-   **Frontend:** Streamlit
-   **LLM & RAG Framework:** LangChain
-   **Language Model:** Google Gemini Pro
-   **Vector Store:** ChromaDB
-   **Embeddings:** Hugging Face Sentence Transformers (`all-MiniLM-L6-v2`)
-   **PDF Processing:** PyPDF



## Local Setup and Installation

Follow these steps to run the application on your local machine.

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/ai-research-chatapp.git](https://github.com/your-username/ai-research-chatapp.git)
cd ai-research-chatapp
```

### 2. Create and Activate a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
# Create the environment
python3 -m venv venv

# Activate the environment
source venv/bin/activate
```

### 3. Install Dependencies

Install all the required Python libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Set Up API Key

The application requires a Google AI API key to use the Gemini model.

1.  Create a folder named `.streamlit` in the root of the project directory.
2.  Inside this folder, create a file named `secrets.toml`.
3.  Add your Google API key to the `secrets.toml` file in the following format:

    ```toml
    GOOGLE_API_KEY = "YOUR_API_KEY_HERE"
    ```

### 5. Run the Application

Once the setup is complete, run the Streamlit application from your terminal.

```bash
streamlit run app.py
```

Your web browser should automatically open a new tab with the running application.

## How to Use

Simply type a question about the research paper into the chat input box at the bottom of the page and press Enter. The assistant will retrieve relevant information from the document and generate a response.