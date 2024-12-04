import streamlit as st
from transformers import pipeline

# Charger les modèles
distilbert_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
bert_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Titre de l'application
st.title("Sentiment Analysis App")
st.write("Compare the results of two models: DistilBERT and BERT")

# Entrée utilisateur
user_input = st.text_area("Enter a sentence to analyze:", "I love using this app!")

if st.button("Analyze Sentiment"):
    # Analyse avec DistilBERT
    distilbert_result = distilbert_analyzer(user_input)[0]
    # Analyse avec BERT
    bert_result = bert_analyzer(user_input)[0]

    # Afficher les résultats
    st.subheader("Results")
    st.write("**DistilBERT:**")
    st.write(f"Sentiment: {distilbert_result['label']}, Confidence: {distilbert_result['score']:.2f}")

    st.write("**BERT:**")
    st.write(f"Sentiment: {bert_result['label']}, Confidence: {bert_result['score']:.2f}")
