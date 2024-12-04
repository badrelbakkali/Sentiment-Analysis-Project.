from transformers import pipeline

# Modèles à comparer
models = {
    "DistilBERT": "distilbert-base-uncased-finetuned-sst-2-english",
    "BERT": "nlptown/bert-base-multilingual-uncased-sentiment"
}

# Liste de phrases personnalisées
custom_texts = [
    "The product quality is amazing, I absolutely love it!",
    "I had to wait 30 minutes for my order, very disappointed.",
    "The movie was okay, but not as great as I expected.",
    "Customer service was excellent, very helpful staff.",
    "This app is a complete waste of time."
]

# Comparer les résultats pour chaque modèle
for model_name, model_path in models.items():
    print(f"Results using {model_name} model:")
    sentiment_analyzer = pipeline("sentiment-analysis", model=model_path)
    for text in custom_texts:
        result = sentiment_analyzer(text)
        print(f"Text: {text}")
        print(f"Sentiment: {result[0]['label']}, Confidence: {result[0]['score']:.2f}")
    print()
