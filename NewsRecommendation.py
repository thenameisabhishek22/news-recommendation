import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import gradio as gr
import plotly.express as px

def load_data():
    news_data = pd.read_csv("News.csv")
    return news_data

news_data = load_data()
categories = news_data["News Category"].value_counts()
label = categories.index
counts = categories.values
figure = px.bar(news_data, x=label,
                y = counts,
            title="Types of News Categories")
figure.show()

def preprocess_text(news_data):
    news_data['Title'] = news_data['Title'].fillna('')
    news_data['Summary'] = news_data['Summary'].fillna('')
    news_data['Combined_Text'] = news_data['Title'] + ' ' + news_data['Summary']
    vectorizer = TfidfVectorizer(stop_words='english')
    news_vectors = vectorizer.fit_transform(news_data['Combined_Text'])
    return news_vectors, vectorizer

def recommend_based_on_keywords(keywords, news_vectors, news_data, vectorizer):
    query_vector = vectorizer.transform([keywords])
    similarity_scores = cosine_similarity(query_vector, news_vectors).flatten()
    top_indices = similarity_scores.argsort()[-20:][::-1]

    recommendations = news_data.iloc[top_indices].copy()
    recommendations['Similarity Score'] = similarity_scores[top_indices]
    return recommendations

def format_recommendations(recommendations):
    result = "**Recommended News Articles:**\n"
    for _, row in recommendations.iterrows():
        result += f"ID: {row['ID']}\nTitle: {row['Title']}\nSummary: {row['Summary']}\n"
        result += f"News Category: {row['News Category']}\nSimilarity Score: {row['Similarity Score']:.2f}\n"
        result += "-" * 50 + "\n"
    return result

def main():
    news_data = load_data()
    news_vectors, vectorizer = preprocess_text(news_data)

    def recommend(user_query):
        recommendations = recommend_based_on_keywords(user_query, news_vectors, news_data, vectorizer)
        return format_recommendations(recommendations)

    iface = gr.Interface(
        fn=recommend,
        inputs="text",
        outputs="text",
        title="News Recommendation System"
    )

    iface.launch()

if __name__ == "__main__":
    main()
    app.run(host="0.0.0.0", port=5000)
