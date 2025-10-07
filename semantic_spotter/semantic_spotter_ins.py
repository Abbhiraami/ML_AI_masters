"""
Libraray_name: Semantic_spotter
Author_name: Abbhiraami S
Description: Functions for Policy information extraction - Functions necessary for building a 
RAG system to assistant policy-holders or insurance agents to compare against policies

"""

### Install libraries
# import subprocess, sys

# packages = [
#     "llama-index",
#     "openai",
#     "llama-index-core",
#     "llama-index-embeddings-openai",
#     "llama-index-embeddings-huggingface",
#     "llama-index-embeddings-cohere",
#     "pymupdf"
# ]

# for pkg in packages:
#     subprocess.run([sys.executable, "-m", "pip", "install", pkg])

### General imports
import os 
import json
import openai
import pandas as pd
## GenAI framework

import nest_asyncio
nest_asyncio.apply()

### LlamaIndex imports

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from functools import lru_cache
import hashlib
import json
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import download_loader
from llama_index.readers.file import PyMuPDFReader 

### Visualization
from IPython.display import display, HTML

### Sentence Chunking - Sentence Splitter
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.text_splitter import TokenTextSplitter
### Section-wise chunking
# from llama_index.core.node_parser import SectionSplitter
from llama_index.core.node_parser import HierarchicalNodeParser

## Semantic Chunking
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.cohere import CohereEmbedding

### Query Engine imports & Response Synthesizer
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core.indices import SummaryIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer


## Import libraries for Retreiving and response generation
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.prompts import PromptTemplate

from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

# Set global default LLM for all RetrieverQueryEngines
with open("C:/Users/SHAMBHAVVISEN/Downloads/OpenAI_API_Key.txt", "r") as f:
    api_key = f.read().strip()

os.environ["OPENAI_API_KEY"] = api_key
print("[INFO] OpenAI API key loaded successfully.")

Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0,api_key=api_key)
print(f"[INFO] Global LLM set to: {Settings.llm.model}")



#### Extraction of documents #####

def load_docs(file_path):
    """
    This function is for loading various documents provided the document folder
    Input: file_path - location for extraction
    Output: documents
    """
    loader=PyMuPDFReader()
    documents=[]

    for file_name in os.listdir(file_path):
        full_name=os.path.join(file_path,file_name)
        docs=loader.load_data(full_name)
        # print(docs.text)
        documents.extend(docs)
    print(f"Loaded {len(documents)}")

    return documents

##### Chunking Strategies ######

def chunking_strategies(chunking_type, documents, chunk_size=512, chunk_overlap=20):
    if chunking_type=="fixed_window":
        if chunk_size is not None or chunk_overlap is not None:
            parser=SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            nodes=parser.get_nodes_from_documents(documents)
        else: 
            parser=SentenceSplitter()
            nodes=parser.get_nodes_from_documents(documents)
    elif chunking_type=="semantic":
        parser=SemanticSplitterNodeParser(
            embed_model=OpenAIEmbedding(),
            similarity_threshold=0.85,
            # settings=settings
        )
        nodes=parser.get_nodes_from_documents(documents)
    elif chunking_type == "Hierarchical":

        # Step 1: Define splitters per level
        level1_splitter = SentenceSplitter(chunk_size=chunk_size[0], chunk_overlap=chunk_overlap[0])
        level2_splitter = SentenceSplitter(chunk_size=chunk_size[1], chunk_overlap=chunk_overlap[1])
        level3_splitter = SentenceSplitter(chunk_size=chunk_size[2], chunk_overlap=chunk_overlap[2])

        # Step 2: Define parser IDs (order matters!)
        node_parser_ids = ["level1", "level2", "level3"]

        # Step 3: Define node_parser_map
        node_parser_map = {
            "level1": level1_splitter,
            "level2": level2_splitter,
            "level3": level3_splitter
        }

        # Step 4: Initialize HierarchicalNodeParser
        parser = HierarchicalNodeParser(
            node_parser_ids=node_parser_ids,
            node_parser_map=node_parser_map
        )

        # Step 5: Parse documents
        nodes = parser.get_nodes_from_documents(documents)

    else:
        raise ValueError("Invalid chunking type. Choose from 'fixed_window', 'semantic', 'Hierarchical'.")
    return nodes

#### Embedding strategies #####

def embedding_strategies(embedding_type,node_type,nodes, api_file_name):
    """
    embedding_type: OpenAIEmbedding, HuggingFace, Cohere
    nodes: chunked nodes from different strategies
    index: VectorStoreIndex
    api_file_path: provide the api_key
    return: index
    """
    ### Setup the environment for persistent storage for different embedding types
    if embedding_type=="OpenAIEmbedding":
        from llama_index.embeddings.openai import OpenAIEmbedding
        with open(api_file_name, "r") as file:
            os.environ["OPENAI_API_KEY"] = file.read().strip()
        persistent_dir=f"storage_openai_{node_type}/"
        embedding_model=OpenAIEmbedding(model="text-embedding-3-large",
                                        api_key=repr(os.getenv("OPENAI_API_KEY")))
    elif embedding_type=="HuggingFace":
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        persistent_dir=f"storage_huggingface_{node_type}/"
        embedding_model=HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    elif embedding_type=="Cohere":
        from llama_index.embeddings.cohere import CohereEmbedding
        with open(api_file_name, "r") as file:
            os.environ["COHERE_API_KEY"] = file.read().strip()
            # print("COHERE_API_KEY:", repr(os.getenv("COHERE_API_KEY"))) 
        persistent_dir=f"storage_cohere_{node_type}/"
        embedding_model = CohereEmbedding(
                    model="embed-english-v3.0",
                    api_key=repr(os.getenv("COHERE_API_KEY"))
        )
    
    print(f"Using embedding_type & embedding model:{embedding_model}")

    if not os.path.exists(persistent_dir):
            print(f"Creating new index... & {embedding_type}")
            os.makedirs(persistent_dir)
            index = VectorStoreIndex.from_documents(nodes,
                                                        embedding=embedding_model,
                                                        persist_dir=persistent_dir)
            ### Store for future use
            index.storage_context.persist(persist_dir=persistent_dir)
            print(f"Done... & {embedding_type}")
            return index
    else:
            ### loading from the existing directory
            print(f"Accessing from existing index...{embedding_type}")
            storage_context=StorageContext.from_defaults(persist_dir=persistent_dir)
            index=load_index_from_storage(storage_context=storage_context)
            print(f"Done... & {embedding_type}")
            return index
    return index


#### Retreiver & Generation #####


# Define cache outside so it persists across calls
_retriever_cache = {}

def create_retriever_generate_response(
        index, 
        query_str="None",
        top_k=3,
        filters=None, 
        node_p="similarity",
        response_mod="tree_summarize",
        cut_off=0.8):
    """
    index: embedding 
    query_str: User_query
    top_k: Matched sets
    filters: Metadata filters
    noe_p: Node postproessors
    response_mod: response format
    cut_off: similarity threshold 
    """

    # ----- STEP 1: Generate a cache key -----
    def make_cache_key():
        index_name = getattr(index, "index_id", str(type(index)))
        key_data = {
            "index": index_name,
            "query": query_str,
            "top_k": top_k,
            "filters": str(filters),
            "node_p": node_p,
            "response_mod": response_mod,
            "cut_off": cut_off
        }
        key_json = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_json.encode()).hexdigest()

    cache_key = make_cache_key()

    # ----- STEP 2: Check cache -----
    if cache_key in _retriever_cache:
        print(f"[CACHE HIT] Returning cached result for query: '{query_str}'")
        return _retriever_cache[cache_key]

    print(f"[CACHE MISS] Processing new query: '{query_str}'")

    # ----- STEP 3: Normal processing -----
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
        filters=filters
    )

    nodes = retriever.retrieve(query_str)
    context_str = "\n\n".join([n.node.get_content() for n in nodes])

    if node_p == "similarity":
        node_postprocessor = SimilarityPostprocessor(similarity_cutoff=cut_off)
    elif node_p == "LLM":
        from llama_index.core.postprocessor.llm_rerank import LLMRerank
        node_postprocessor = LLMRerank(
            llm=OpenAI(model="gpt-4o-mini", temperature=0),
            top_n=3
        )
    else:
        node_postprocessor = None

    comparison_keywords = ["compare", "difference", "vs", "versus", "differ", "contrast", "between"]
    is_comparison_query = any(word in query_str.lower() for word in comparison_keywords)

    if is_comparison_query:
        prompt_text = (
            "You are an expert insurance policy analyst.\n"
            "Return your answer in a **Markdown table**:\n\n"
            "| Policy Name | Feature / Benefit | Details |\n"
            "|--------------|------------------|----------|\n"
            "Populate with key distinctions only.\n\n"
            "Query: {query_str}\n"
            "Context: {context_str}\n"
            "Answer:"
        )
    else:
        prompt_text = (
            "You are an insurance expert. Provide a detailed, structured response.\n\n"
            "Query: {query_str}\n"
            "Context: {context_str}\n"
            "Answer:"
        )

    custom_prompt = PromptTemplate(prompt_text)

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[node_postprocessor] if node_postprocessor else [],
        response_synthesizer=get_response_synthesizer(
            response_mode=response_mod,
            text_qa_template=custom_prompt
        )
    )

    response = query_engine.query(query_str)

    # ----- STEP 4: Store in cache -----
    result_tuple = (query_engine, nodes, context_str, response.response)
    _retriever_cache[cache_key] = result_tuple

    print(f"[CACHE STORE] Query cached successfully: '{query_str}'")

    return result_tuple


# def create_retriever_generate_response(
#         index, 
#         query_str="None",
#         top_k=3,
#         filters=None, 
#         node_p="similarity",
#         response_mod="tree_summarize",
#         cut_off=0.8):
#     """
#     index: embedding 
#     query_str: User_query
#     top_k: Matched sets
#     filters: Metadata filters
#     noe_p: Node postproessors
#     response_mod: response format
#     cut_off: similarity threshold 
#     """

#     # Create retriever
#     retriever = VectorIndexRetriever(
#         index=index,
#         similarity_top_k=top_k,
#         filters=filters
#     )

#     # Retrieve nodes and build context string
#     nodes = retriever.retrieve(query_str)
#     context_str = "\n\n".join([n.node.get_content() for n in nodes])

#     # Postprocessor
#     if node_p == "similarity":
#         node_postprocessor = SimilarityPostprocessor(similarity_cutoff=cut_off)
#     elif node_p == "LLM":
#         from llama_index.core.postprocessor.llm_rerank import LLMRerank
#         node_postprocessor = LLMRerank(
#             llm=OpenAI(model="gpt-4o-mini",temperature=0),
#             top_n=3
#         )
#     else:
#         node_postprocessor = None

#     # Prompt
#     comparison_keywords = ["compare", "difference", "vs", "versus", "differ", "contrast", "between"]
#     is_comparison_query = any(word in query_str.lower() for word in comparison_keywords)
#     if is_comparison_query:

#         prompt_text = (
#             "You are an expert insurance policy analyst.\n"
#             "You MUST return your answer in a clean **Markdown table** format with these columns:\n\n"
#             "| Policy Name | Feature / Benefit | Details |\n"
#             "|--------------|------------------|----------|\n"
#             "Populate the table with key distinctions and benefits clearly.\n"
#             "Do NOT add any text outside the table.\n\n"
#             "Query: {query_str}\n"
#             "Context: {context_str}\n"
#             "Answer:"
#         )
#     else:
#         prompt_text = (
#         "You are an insurance expert. Provide a detailed, structured response "
#         "based on the context extracted from policy documents.\n\n"
#         "Query: {query_str}\n"
#         "Context: {context_str}\n"
#         "Answer:"
#     )

#     custom_prompt = PromptTemplate(prompt_text)


#     # Query engine
#     query_engine = RetrieverQueryEngine(
#         retriever=retriever,
#         node_postprocessors=[node_postprocessor] if node_postprocessor else [],
#         response_synthesizer=get_response_synthesizer(
#             response_mode=response_mod,
#             text_qa_template=custom_prompt
#         )
#     )

#     response = query_engine.query(query_str)
#     return query_engine,nodes, context_str, response.response

