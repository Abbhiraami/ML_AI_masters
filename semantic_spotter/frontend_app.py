import streamlit as st
import re
import pandas as pd
from semantic_spotter_ins import (
    load_docs,
    chunking_strategies,
    embedding_strategies,
    create_retriever_generate_response
)

st.set_page_config(page_title="Insurance Policy Assistant", layout="wide")

st.title("🧾 Insurance Policy Semantic Assistant")

# --- Sidebar options ---
st.sidebar.header("Configuration")

# Folder path
file_loc = st.sidebar.text_input("📁 Folder path for policy documents", "Policy+Documents")

# Chunking options
chunking_type = st.sidebar.selectbox(
    "🔪 Chunking Type", 
    ["fixed_window", "semantic", "Hierarchical"]
)

if chunking_type == "Hierarchical":
    chunk_size = st.sidebar.text_input("Chunk sizes (comma-separated)", "512,256,128")
    chunk_overlap = st.sidebar.text_input("Chunk overlaps (comma-separated)", "50,25,10")
    chunk_size = [int(x) for x in chunk_size.split(",")]
    chunk_overlap = [int(x) for x in chunk_overlap.split(",")]
else:
    chunk_size = st.sidebar.number_input("Chunk Size", value=512)
    chunk_overlap = st.sidebar.number_input("Chunk Overlap", value=50)

# Embedding options
embedding_type = st.sidebar.selectbox(
    "🧬 Embedding Type", 
    ["OpenAIEmbedding", "HuggingFace", "Cohere"]
)
api_file_name = st.sidebar.text_input("🔑 API Key File Path", "C:/Users/SHAMBHAVVISEN/Downloads/OpenAI_API_Key.txt")

# Response mode
response_mode = st.sidebar.selectbox(
    "💬 Response Mode", 
    ["tree_summarize", "refine", "compact"]
)

# --- User Query ---
st.subheader("Ask your question:")
query_str = st.text_area("Enter your policy-related question here")

# --- Process button ---
if st.button("Run Semantic Assistant"):
    with st.spinner("🔍 Loading documents and building response..."):
        try:
            # Step 1: Load documents
            documents = load_docs(file_loc)
            
            # Step 2: Chunking
            nodes = chunking_strategies(chunking_type, documents, chunk_size, chunk_overlap)

            # Step 3: Embedding
            index = embedding_strategies(embedding_type, chunking_type, nodes, api_file_name)

            # Step 4: Generate Response
            _, _, _, response = create_retriever_generate_response(
                index=index,
                query_str=query_str,
                response_mod=response_mode
            )

            
            def markdown_table_to_df(md_text):

                # Extract markdown table lines
                lines = [line.strip() for line in md_text.split("\n") if "|" in line]
                if len(lines) < 2:
                    return None
                # Remove alignment line (the --- one)
                header = [h.strip() for h in lines[0].split("|")[1:-1]]
                data_lines = lines[2:] if "---" in lines[1] else lines[1:]
                rows = [[c.strip() for c in row.split("|")[1:-1]] for row in data_lines]
                return pd.DataFrame(rows, columns=header)

            ...

            st.success("✅ Response generated successfully!")

            # Try to parse as table
            if "|" in response:
                df = markdown_table_to_df(response)
                if df is not None:
                    st.markdown("### 📊 Comparison Table")
                    st.dataframe(df, use_container_width=True)
                else:
                    st.markdown(response, unsafe_allow_html=True)
            else:
                st.markdown(response, unsafe_allow_html=True)

        
        except Exception as e:
            st.error(f"Error: {e}")

