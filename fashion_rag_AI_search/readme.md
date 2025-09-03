# 👗 Fashion AI – Personal Stylist (Outfit Builder with RAG)  

This repository demonstrates an **AI-powered Outfit Builder** that uses **Retrieval-Augmented Generation (RAG)** to recommend complementary fashion items (cross-sell) from the [Myntra Fashion Dataset](https://www.kaggle.com/datasets/djagatiya/myntra-fashion-product-dataset).  

---

## 📌 Problem Statement  
Finding the perfect outfit combination can be overwhelming with large fashion catalogs. This project acts as a **Personal Stylist AI**, answering queries like:  
> *“What goes well with these blue jeans?”*  

The system retrieves relevant items (tops, shoes, accessories) and generates styled outfit suggestions.  

---

## 🚀 Features  
- **Outfit Builder (Cross-sell):** Suggests complementary products for a given item.  
- **Semantic Search:** Embeds product text + attributes for similarity retrieval.  
- **Query Parsing:** Extracts color, style, material, pattern, occasion, price, and brand.  
- **Re-ranking Layer:** Improves relevance using cross-encoder models.  
- **Generative Layer:** Produces human-like outfit suggestions.  
- **Caching Layer:** Speeds up repeated queries.  

---

## 🛠️ Workflow  
1. **Data Preparation** – Clean product metadata & descriptions.  
2. **Embedding Layer** – Generate embeddings using `text-embedding-ada-002`.  
3. **Vector Store** – Store & query embeddings in **ChromaDB**.  
4. **Query Parsing** – LLM extracts filters (color, occasion, price, brand).  
5. **Retrieval** – Fetch top-N candidates via semantic similarity.  
6. **Re-ranking** – Cross-encoder refines the ranking.  
7. **Generation** – LLM suggests styled outfits in natural language.  

