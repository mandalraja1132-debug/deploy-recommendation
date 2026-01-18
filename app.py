from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load lightweight CSV
movies = pd.read_csv("movies_processed.csv")

# Build vectors (this runs once)
cv = CountVectorizer(max_features=2000, stop_words="english")
vectors = cv.fit_transform(movies["tags"]).toarray()

# Compute similarity matrix (smaller, manageable)
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

@app.get("/")
def home():
    return "Movie Recommendation API is running"

@app.get("/recommend")
def recommend_movie(movie: str):
    return recommend(movie)
