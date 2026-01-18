import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🎬 Movie Recommendation System")

# Load data
movies = pd.read_csv("movies_processed.csv")

# Build vectors
cv = CountVectorizer(max_features=2000, stop_words="english")
vectors = cv.fit_transform(movies["tags"]).toarray()
similarity = cosine_similarity(vectors)

def recommend(movie):
    movie = movie.lower()
    movies["title_lower"] = movies["title"].str.lower()

    if movie not in movies["title_lower"].values:
        return ["Movie not found"]

    index = movies[movies["title_lower"] == movie].index[0]
    distances = similarity[index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:4]

    return [movies.iloc[i[0]].title for i in movies_list]

# User input
movie_name = st.text_input("Enter a movie name")

if st.button("Recommend"):
    recommendations = recommend(movie_name)

    st.subheader("Recommended Movies:")
    for movie in recommendations:
        st.write(movie)
