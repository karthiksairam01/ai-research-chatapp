import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


@st.cache_resource
def load_and_process_pdf():
    """Loads a PDF, splits it into chunks, and creates a vector store."""
    loader = PyPDFLoader("data/Paraphrase_generation.pdf") 
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    docs = text_splitter.split_documents(documents)

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 7})

@st.cache_resource
def get_rag_chain():
    """Creates the RAG chain using a cached LLM and prompt."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=st.secrets["GOOGLE_API_KEY"])

    prompt = ChatPromptTemplate.from_template("""
    Synthesize a concise answer to the following question based ONLY on the provided context.
    Combine information from the different parts of the context to create a coherent summary.
    If the answer cannot be formed from the context, say that the information is not available in the document.

    <context>
    {context}
    </context>

    Question: {input}
    """)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = load_and_process_pdf()
    return create_retrieval_chain(retriever, document_chain)


# --- Streamlit App Interface ---

st.set_page_config(page_title="Chat with Karthik's Research", layout="wide")
st.title("Chat with Karthik's NLP Research 📄")
st.write("Ask a question about my research paper on Paraphrase Detection in Kannada.")

# Get the RAG chain
rag_chain = get_rag_chain()

# Main chat interface
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the paper..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"input": prompt})
            st.markdown(response["answer"])
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})