# Jarvis - Personal AI Assistant

This project is a simple implementation of a personal AI assistant similar to Jarvis.
The goal of the project is to understand how a self-hosted language model can be
combined with a vector database to answer user queries.

## Technologies Used
- Python
- LLaMA (self-hosted using Ollama)
- Pinecone (vector database)
- Streamlit (chat interface)

## How It Works
1. Documents are converted into vector embeddings and stored in Pinecone.
2. When the user asks a question, the query is also converted into an embedding.
3. The most relevant stored information is retrieved from Pinecone.
4. This information is passed to the LLaMA model to generate a response.

## How to Run
1. Install the dependencies using:
   pip install -r requirements.txt

2. Start Ollama and make sure the LLaMA model is running.

3. Run the ingestion script:
   python ingest.py

4. Start the Streamlit app:
   streamlit run app.py

## Note
This project is built for learning purposes and focuses on simplicity rather than
production-level optimization.
