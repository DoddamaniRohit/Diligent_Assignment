import streamlit as st
import pinecone
from sentence_transformers import SentenceTransformer
from llm_helper import ask_llama

st.title(" Jarvis - Personal AI Assistant")

pinecone.init(
    api_key="YOUR_PINECONE_API_KEY",
    environment="YOUR_ENV"
)

index = pinecone.Index("jarvis-index")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

user_query = st.text_input("Ask me something:")

if user_query:
    query_vector = embedder.encode(user_query).tolist()
    result = index.query(vector=query_vector, top_k=1, include_metadata=True)

    context = ""
    if result["matches"]:
        context = result["matches"][0]["metadata"]["text"]

    prompt = f"""
    Use the following information to answer the question.

    Context: {context}

    Question: {user_query}
    """

    answer = ask_llama(prompt)
    st.write("**Jarvis:**", answer)
