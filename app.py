from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the Excel file into a pandas DataFrame
data = pd.read_excel('DATABASE.xls')

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['query_search'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']

    # Transform the user's query into a TF-IDF vector
    query_vector = vectorizer.transform([query])

    # Calculate the cosine similarity between the user's query vector and the TF-IDF matrix
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Get the indices of the most similar queries
    closest_indices = similarity_scores.argsort()[::-1][:5]  # Adjust the number of results to display

    # Retrieve the closest queries
    closest_queries = data['query_search'].iloc[closest_indices]

    return render_template('result.html', query=query, queries=closest_queries)

if __name__ == '__main__':
    app.run(debug=True)
