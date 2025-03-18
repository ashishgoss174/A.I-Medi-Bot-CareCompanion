# A.I-Medi-Bot-CareCompanion

## Overview:

Ask CareCompanion! is an AI-powered medical chatbot that provides answers based on retrieved medical knowledge. It uses retrieval-augmented generation (RAG) to fetch relevant information before generating responses.

## How It Works: 

The user asks a medical question.
The chatbot retrieves relevant medical documents using a vector database (FAISS).
The retrieved documents are passed to a language model (Mistral-7B) to generate a response.
The chatbot returns an answer along with sources.

## Models Used 🧠

1️. Mistral-7B-Instruct-v0.3 (Language Model)
Used for: Generating responses based on retrieved context.
Why? A powerful instruction-tuned model that understands and answers medical queries accurately.
Source: Mistral-7B-Instruct-v0.3 on Hugging Face

2️. all-MiniLM-L6-v2 (Embedding Model)
Used for: Converting text into numerical vectors for semantic search in FAISS.
Why? Efficient and lightweight, optimized for similarity search.
Source: all-MiniLM-L6-v2 on Hugging Face

## Prerequisite:

Install Anaconda

### Install Required Packages:

pip install langchain langchain_community langchain_huggingface faiss-cpu pypdf
pip install huggingface_hub
pip install streamlit

Steps to run:
1. Run create_memory_for_llm.py file.

2. login to hugging face website and paste "mistralai/Mistral-7B-Instruct-v0.3", select this model and generate new token and copy that token ID.(Note: We must not share .env file with anyone).

3. create a .env file and write => HF_TOKEN="COPIED TOKEN ID".

4. Run connect_memory_with_llm.py file.

5. Run carecompanion.py file for UI.
