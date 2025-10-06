# 🧾 Semantic Spotter – Insurance Policy Assistant
---

## 👩‍💻 Author

**Abbhiraami S**  

---

> This project was developed as part of ongoing research on the application of GenAI and Retrieval-Augmented Generation (RAG) systems in the insurance and financial compliance domain.

---

## 🧠 General Introduction

**Semantic Spotter** is an intelligent **Retrieval-Augmented Generation (RAG)** application designed to help insurance agents, analysts, and policyholders **query, compare, and analyze insurance policy documents** using natural language.  

It leverages **LlamaIndex**, **OpenAI embeddings**, and **Streamlit** to extract relevant information from policy PDFs, organize them into meaningful chunks, and generate structured answers or comparison tables using LLMs.

The goal of this project is to simplify insurance document understanding and policy comparison through a single conversational interface powered by GenAI.

---

## 🧰 Tech Stack

| Component | Library / Tool | Purpose |
|------------|----------------|----------|
| **Frontend** | Streamlit | Interactive web interface for querying |
| **Backend Framework** | LlamaIndex | RAG pipeline for retrieval and context management |
| **Language Model** | OpenAI GPT models | Response generation and summarization |
| **Embeddings** | OpenAI / HuggingFace / Cohere | Text vectorization for semantic search |
| **Document Loader** | PyMuPDF | Extracts text content from PDF policy documents |
| **Data Handling** | Pandas | Tabular representation for comparison results |
| **Caching** | Python Dictionary | In-memory cache for faster repeated queries |
| **Async Handling** | nest_asyncio | Enables async operations within Streamlit |
| **Environment** | Python 3.9+ | Compatible runtime for all dependencies |

---

## 📁 Project Structure

│
├── frontend_app.py # Streamlit frontend application
├── semantic_spotter_ins.py # Backend RAG functions (document loading, chunking, embedding, retrieval)
├── insure_assist.ipynb # Debugging and evaluation notebook
├── Policy+Documents/ # Folder containing insurance policy PDFs
└── README.md # Project documentation


> 🧪 **Note:** The `.ipynb` file (`insure_assist.ipynb`) is provided **for debugging and evaluation purposes only**.  
> It can be used to test individual functions, tune chunking logic, or validate retrieval results before deployment.

---

## 🧩 Steps to Use the Application

### 1️⃣ Launch the App
Run the Streamlit command below to start the application.  
You’ll see the **Semantic Spotter** interface open in your browser.

```bash
streamlit run frontend_app.py
```
## 2️⃣ Configure Parameters in the Sidebar

📁 Folder Path → Path to your policy documents (e.g., Policy+Documents)

🔪 Chunking Type → Choose between fixed_window, semantic, or Hierarchical

🧩 Chunk Size and Overlap → Define how text is split (applies to fixed or hierarchical)

🧬 Embedding Type → Select OpenAIEmbedding, HuggingFace, or Cohere

💬 Response Mode → Choose response summarization style (tree_summarize, refine, compact)

3️⃣ Enter Your Query

Type your question into the main input box.

Example queries:

“What is the death benefit in HDFC Life Sanchay Plus?”

“Compare surrender value between LIC Jeevan Labh and HDFC Sanchay Plus.”

“Explain maturity benefits under Max Life Guaranteed Savings Plan.”

4️⃣ Click ‘Run Semantic Assistant’

The system will automatically:

Load policy documents

Apply the selected chunking and embedding strategy

Retrieve relevant context using semantic similarity

Generate an appropriate response (structured or tabular)

5️⃣ View the Output

Single policy queries → Displayed as a detailed structured explanation

Comparison queries → Rendered as a formatted Markdown table directly in Streamlit

